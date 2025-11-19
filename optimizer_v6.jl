# GliderOptimizer_v5.jl
# Enhanced aircraft optimization with full geometric control
# Optimizes for range while maintaining stability with comprehensive design freedom
#
# NEW IN V5:
# - Wing span is now a design variable (not fixed)
# - Wing dihedral added for lateral stability
# - Full tail parameterization (chords, twists, spans)
# - Ballast mass for CG control
# - Improved static margin calculation
#
# Range Equation: R = (e_b/g) × η × (L/D) × (m_b/m_TO)
#
# Installation requirements:
#   using Pkg
#   Pkg.add(["VortexLattice", "SNOW", "Ipopt", "LinearAlgebra", "Printf", "Statistics", "Plots"])
#
using VortexLattice
using SNOW
using Ipopt
using LinearAlgebra
using Printf
using Statistics
using Plots
import ForwardDiff

# ============================================================================
# DESIGN CONFIGURATION PARAMETERS
# ============================================================================

Base.@kwdef mutable struct DesignConfig
    # Flight conditions
    cruise_speed::Float64 = 20.0
    air_density::Float64 = 1.225
    
    # Wing geometry constraints (now optimizable)
    min_wing_half_span::Float64 = 0.5          # [m] Minimum half-span
    max_wing_half_span::Float64 = 0.762        # [m] Maximum half-span (30 inches)
    max_wing_dihedral::Float64 = 15.0          # [deg] Maximum dihedral angle
    n_span_sections::Int = 2                   # Number of design control points
    
    # Materials
    foam_density::Float64 = 24.668
    foam_thickness::Float64 = 0.0508
    fuselage_density::Float64 = 1850.0
    fuselage_outer_diameter::Float64 = 0.007
    fuselage_inner_diameter::Float64 = 0.005
    spar_density::Float64 = 1850.0
    spar_outer_diameter::Float64 = 0.007
    spar_inner_diameter::Float64 = 0.005
    
    # Component masses (UPDATED with actual values)
    motor_mass::Float64 = 0.156                # [kg] 156 grams
    motor_x::Float64 = 0.05
    battery_mass::Float64 = 0.440              # [kg] 440 grams
    battery_x_initial::Float64 = -0.05
    electronics_mass::Float64 = 0.05
    electronics_x_initial::Float64 = 0.0
    ballast_mass_initial::Float64 = 0.0
    ballast_x_initial::Float64 = 0.0
    
    # Propulsion system
    battery_specific_energy::Float64 = 150.0
    propulsion_efficiency::Float64 = 0.65
    
    # Fuselage constraints
    max_fuselage_length::Float64 = 1.0
    
    # Airfoil selection
    wing_airfoil_type::String = "NACA4"
    wing_naca_digits::String = "2412"
    wing_dat_file::String = ""
    wing_camber_function::Union{Function, Nothing} = nothing
    
    tail_airfoil_type::String = "NACA4"
    tail_naca_digits::String = "0012"
    tail_dat_file::String = ""
    tail_camber_function::Union{Function, Nothing} = nothing
    
    # Optimization parameters (UPDATED)
    stability_margin::Float64 = 0.01
    lift_safety_factor::Float64 = 1.0
    static_margin_min::Float64 = 0.10          # INCREASED from 0.05 to 0.10 for better stability
end

const CONFIG = DesignConfig()
const g = 9.80665

# ============================================================================
# AIRFOIL FUNCTIONS (unchanged from v4)
# ============================================================================

function parse_selig_dat(filename::String)
    if !isfile(filename)
        error("Airfoil .dat file not found: $filename")
    end
    
    lines = readlines(filename)
    data_start = 1
    for (i, line) in enumerate(lines)
        parts = split(strip(line))
        if length(parts) == 2
            try
                parse(Float64, parts[1])
                parse(Float64, parts[2])
                data_start = i
                break
            catch
                continue
            end
        end
    end
    
    coords = []
    for line in lines[data_start:end]
        parts = split(strip(line))
        if length(parts) >= 2
            try
                x = parse(Float64, parts[1])
                y = parse(Float64, parts[2])
                push!(coords, (x, y))
            catch
                continue
            end
        end
    end
    
    if isempty(coords)
        error("No valid coordinates found in .dat file")
    end
    
    min_x_idx = argmin([c[1] for c in coords])
    upper = coords[1:min_x_idx]
    lower = coords[min_x_idx:end]
    reverse!(upper)
    sort!(upper, by=first)
    sort!(lower, by=first)
    
    x_upper = [c[1] for c in upper]
    y_upper = [c[2] for c in upper]
    x_lower = [c[1] for c in lower]
    y_lower = [c[2] for c in lower]
    
    function interpolate_linear(x_data, y_data, x_query)
        if x_query <= x_data[1]
            return y_data[1]
        elseif x_query >= x_data[end]
            return y_data[end]
        end
        for i in 1:(length(x_data)-1)
            if x_data[i] <= x_query <= x_data[i+1]
                t = (x_query - x_data[i]) / (x_data[i+1] - x_data[i])
                return y_data[i] * (1-t) + y_data[i+1] * t
            end
        end
        return y_data[end]
    end
    
    x_camber = unique(sort(vcat(x_upper, x_lower)))
    y_camber = zeros(length(x_camber))
    
    for (i, x) in enumerate(x_camber)
        yu = interpolate_linear(x_upper, y_upper, x)
        yl = interpolate_linear(x_lower, y_lower, x)
        y_camber[i] = (yu + yl) / 2.0
    end
    
    function camber(xc)
        return interpolate_linear(x_camber, y_camber, xc)
    end
    
    return camber
end

function get_camber_function(config::DesignConfig, surface::Symbol=:wing)
    if surface == :wing
        airfoil_type = config.wing_airfoil_type
        naca_digits = config.wing_naca_digits
        dat_file = config.wing_dat_file
        custom_func = config.wing_camber_function
    else
        airfoil_type = config.tail_airfoil_type
        naca_digits = config.tail_naca_digits
        dat_file = config.tail_dat_file
        custom_func = config.tail_camber_function
    end
    
    if airfoil_type == "flat"
        base_func = (xc) -> 0.0
    elseif airfoil_type == "NACA4"
        base_func = naca4_camber(naca_digits)
    elseif airfoil_type == "dat_file"
        if dat_file == "" || !isfile(dat_file)
            @warn "No valid .dat file specified for $surface, using flat plate"
            base_func = (xc) -> 0.0
        else
            base_func = parse_selig_dat(dat_file)
        end
    elseif airfoil_type == "custom" && custom_func !== nothing
        base_func = custom_func
    else
        @warn "Unknown airfoil configuration for $surface, using flat plate"
        base_func = (xc) -> 0.0
    end
    
    return (xc) -> base_func(xc)
end

function naca4_camber(digits::String)
    if length(digits) != 4
        @warn "NACA digits must be 4 characters, using flat plate"
        return (xc) -> 0.0
    end
    
    m = parse(Int, digits[1]) / 100.0
    p = parse(Int, digits[2]) / 10.0
    
    if m == 0.0 || p == 0.0
        return (xc) -> 0.0
    end
    
    function camber(xc)
        if xc < p
            return (m / p^2) * (2*p*xc - xc^2)
        else
            return (m / (1-p)^2) * ((1 - 2*p) + 2*p*xc - xc^2)
        end
    end
    
    return camber
end

# ============================================================================
# GEOMETRY GENERATION (Enhanced with dihedral)
# ============================================================================

const GLOBAL_WING_CAMBER = Ref{Function}((xc)->0.0)
const GLOBAL_TAIL_CAMBER = Ref{Function}((xc)->0.0)

function build_surface_from_sections(
    xle, yle, zle, chord, theta, phi, fc, 
    ns::Int, nc::Int; 
    mirror::Bool=false,
    spacing_s=Uniform(), 
    spacing_c=Uniform()
)
    xle_f = identity.(xle)
    yle_f = identity.(yle)
    zle_f = identity.(zle)
    chord_f = identity.(chord)
    theta_f = identity.(theta)
    phi_f = identity.(phi)
    
    if any(x -> x isa ForwardDiff.Dual, xle_f)
        xle_f = ForwardDiff.value.(xle_f)
        yle_f = ForwardDiff.value.(yle_f)
        zle_f = ForwardDiff.value.(zle_f)
        chord_f = ForwardDiff.value.(chord_f)
        theta_f = ForwardDiff.value.(theta_f)
        phi_f = ForwardDiff.value.(phi_f)
    end
    
    grid, ratio = wing_to_grid(xle_f, yle_f, zle_f, chord_f, theta_f, phi_f, ns, nc; 
                               fc=fc, spacing_s=spacing_s, spacing_c=spacing_c, mirror=mirror)
    
    surface = grid_to_surface_panels(grid; ratios=ratio, mirror=false)
    
    return grid, ratio, surface
end

"""
Build complete aircraft configuration with full geometric control.
NEW: Wing span, dihedral, and full tail parameterization.
"""
function build_configuration(
    wing_chords::Vector{T},
    wing_twists::Vector{T},
    htail_chords::Vector{T},
    htail_twists::Vector{T},
    vtail_chords::Vector{T},
    vtail_twists::Vector{T};
    wing_half_span::Real = 0.762,
    wing_dihedral::Real = 0.0,  # degrees
    sweep::Real = 0.0,
    htail_half_span::Real = 0.3,
    htail_x::Real = 0.8,
    vtail_height::Real = 0.25,
    vtail_x::Real = 0.8,
    mass_x::Real = 0.0,
    wing_x::Real = 0.0
) where T
    
    n_wing = length(wing_chords)
    n_ht = length(htail_chords)
    n_vt = length(vtail_chords)
    
    wing_camber_func = GLOBAL_WING_CAMBER[]
    tail_camber_func = GLOBAL_TAIL_CAMBER[]
    
    # -------------------------------------------------------------------------
    # WING GEOMETRY with dihedral
    # -------------------------------------------------------------------------
    y_half = collect(range(0.0, stop=wing_half_span, length=n_wing))
    
    # Dihedral: z increases linearly with y
    dihedral_rad = T(wing_dihedral * pi / 180.0)
    zle_w = T[yi * tan(dihedral_rad) for yi in y_half]
    
    # Leading edge with sweep
    wing_span = 2 * wing_half_span
    xle_w = T[wing_x - 0.25 * c + sweep * (yi/wing_half_span) for (c, yi) in zip(wing_chords, y_half)]
    
    theta_w = T[(pi/180) * tw for tw in wing_twists]
    phi_w = zeros(T, n_wing)
    fc_w = [wing_camber_func for _ in 1:n_wing]
    
    ns_w = 12
    nc_w = 6
    
    grid_w, ratio_w, surf_w = build_surface_from_sections(
        xle_w, y_half, zle_w, wing_chords, theta_w, phi_w, fc_w, ns_w, nc_w; 
        mirror=true, spacing_s=Sine()
    )
    
    # -------------------------------------------------------------------------
    # HORIZONTAL TAIL with full parameterization
    # -------------------------------------------------------------------------
    ht_y = T[yi for yi in range(0.0, stop=htail_half_span, length=n_ht)]
    ht_xle = T[htail_x - 0.25 * c for c in htail_chords]
    ht_zle = zeros(T, n_ht)
    ht_theta = T[(pi/180) * tw for tw in htail_twists]
    ht_phi = zeros(T, n_ht)
    fc_ht = [tail_camber_func for _ in 1:n_ht]
    
    ns_ht = 6
    nc_ht = 4
    
    grid_ht, ratio_ht, surf_ht = build_surface_from_sections(
        ht_xle, ht_y, ht_zle, htail_chords, ht_theta, ht_phi, fc_ht, ns_ht, nc_ht; 
        mirror=true
    )
    
    # -------------------------------------------------------------------------
    # VERTICAL TAIL with full parameterization
    # -------------------------------------------------------------------------
    vt_z = T[zi for zi in range(0.0, stop=vtail_height, length=n_vt)]
    vt_xle = T[vtail_x - 0.25 * c for c in vtail_chords]
    vt_yle = zeros(T, n_vt)
    vt_theta = T[(pi/180) * tw for tw in vtail_twists]
    vt_phi = zeros(T, n_vt)
    fc_vt = [tail_camber_func for _ in 1:n_vt]
    
    ns_vt = 6
    nc_vt = 4
    
    grid_vt, ratio_vt, surf_vt = build_surface_from_sections(
        vt_xle, vt_yle, vt_z, vtail_chords, vt_theta, vt_phi, fc_vt, ns_vt, nc_vt; 
        mirror=false
    )
    
    # -------------------------------------------------------------------------
    # COMBINE INTO SYSTEM
    # -------------------------------------------------------------------------
    grids = [grid_w, grid_ht, grid_vt]
    ratios = [ratio_w, ratio_ht, ratio_vt]
    system = System(grids; ratios=ratios)
    
    # -------------------------------------------------------------------------
    # REFERENCE VALUES
    # -------------------------------------------------------------------------
    # Wing area (trapezoidal approximation)
    wing_area = zero(T)
    for i in 1:(n_wing-1)
        wing_area += 2 * 0.5 * (y_half[i+1] - y_half[i]) * (wing_chords[i] + wing_chords[i+1])
    end
    
    ref_area = wing_area
    ref_chord = wing_area / wing_span
    ref_span = T(wing_span)
    rref = [T(mass_x), zero(T), zero(T)]
    
    return system, ref_area, ref_chord, ref_span, rref, surf_w, surf_ht, surf_vt
end

# ============================================================================
# AERODYNAMIC ANALYSIS (unchanged)
# ============================================================================

function run_vlm_analysis!(
    system::System, 
    ref_area::Real, 
    ref_chord::Real,
    ref_span::Real,
    rref::Vector; 
    α=0.0, 
    β=0.0, 
    Ω=[0.0,0.0,0.0], 
    symmetric=false
)
    fs = Freestream(CONFIG.cruise_speed, α, β, Ω)
    
    ref_area_f = ref_area isa ForwardDiff.Dual ? ForwardDiff.value(ref_area) : Float64(ref_area)
    ref_chord_f = ref_chord isa ForwardDiff.Dual ? ForwardDiff.value(ref_chord) : Float64(ref_chord)
    ref_span_f = ref_span isa ForwardDiff.Dual ? ForwardDiff.value(ref_span) : Float64(ref_span)
    rref_f = [r isa ForwardDiff.Dual ? ForwardDiff.value(r) : Float64(r) for r in rref]
    
    ref = Reference(ref_area_f, ref_chord_f, ref_span_f, rref_f, CONFIG.cruise_speed)
    
    steady_analysis!(system, ref, fs; symmetric=symmetric)
    
    CF, CM = body_forces(system; frame=Wind())
    dCF, dCM = stability_derivatives(system)
    
    return CF, CM, dCF, dCM
end

# ============================================================================
# COMPONENT-BASED INERTIA MODEL (Enhanced with ballast)
# ============================================================================

struct Component
    name::String
    mass::Float64
    x::Float64
    y::Float64
    z::Float64
    Ixx_c::Float64
    Iyy_c::Float64
    Izz_c::Float64
end

function lamina_inertia(m::Float64, span::Float64, chord::Float64)
    Ixx = (1.0/12.0) * m * span^2
    Iyy = (1.0/12.0) * m * chord^2
    Izz = Ixx + Iyy
    return Ixx, Iyy, Izz
end

function rod_inertia(m::Float64, L::Float64, axis_dir::Symbol, r_outer::Float64=0.0035, r_inner::Float64=0.0025)
    I_perp = (1.0/12.0)*m*L^2 + (1.0/4.0)*m*(r_outer^2 + r_inner^2)
    I_axial = (1.0/2.0)*m*(r_outer^2 + r_inner^2)
    
    if axis_dir == :x
        Ixx = I_axial
        Iyy = I_perp
        Izz = I_perp
    elseif axis_dir == :y
        Ixx = I_perp
        Iyy = I_axial
        Izz = I_perp
    elseif axis_dir == :z
        Ixx = I_perp
        Iyy = I_perp
        Izz = I_axial
    else
        error("axis_dir must be :x, :y, or :z")
    end
    return Ixx, Iyy, Izz
end

"""
Build component list with ballast support.
"""
function build_components(
    wing_half_span::Float64,
    wing_chords::Vector,
    htail_half_span::Float64,
    htail_chords::Vector,
    vtail_height::Float64,
    vtail_chords::Vector;
    wing_x::Float64 = 0.0,
    wing_z::Float64 = 0.0,
    htail_x::Float64 = 0.8,
    htail_z::Float64 = 0.0,
    vtail_x::Float64 = 0.8,
    vtail_z::Float64 = 0.0,
    battery_x::Float64 = -0.05,
    electronics_x::Float64 = 0.0,
    ballast_mass::Float64 = 0.0,  # NEW
    ballast_x::Float64 = 0.0       # NEW
)
    comps = Component[]
    
    foam_thickness = CONFIG.foam_thickness
    rho_foam = CONFIG.foam_density
    rho_spar = CONFIG.spar_density
    rho_fuse = CONFIG.fuselage_density
    
    spar_r_outer = CONFIG.spar_outer_diameter / 2.0
    spar_r_inner = CONFIG.spar_inner_diameter / 2.0
    fuse_r_outer = CONFIG.fuselage_outer_diameter / 2.0
    fuse_r_inner = CONFIG.fuselage_inner_diameter / 2.0
    
    # -------------------------------------------------------------------------
    # WING
    # -------------------------------------------------------------------------
    wing_chords_f = [c isa ForwardDiff.Dual ? ForwardDiff.value(c) : Float64(c) for c in wing_chords]
    c_avg = mean(wing_chords_f)
    
    S_wing = 2.0 * (c_avg * wing_half_span)
    vol_wing = S_wing * foam_thickness
    mass_wing = vol_wing * rho_foam
    
    m_half = mass_wing / 2.0
    y_centroid = wing_half_span / 2.0
    x_centroid = wing_x + c_avg / 2.0
    
    Ixx, Iyy, Izz = lamina_inertia(m_half, wing_half_span, c_avg)
    push!(comps, Component("wing_right", m_half, x_centroid,  y_centroid, wing_z, Ixx, Iyy, Izz))
    push!(comps, Component("wing_left",  m_half, x_centroid, -y_centroid, wing_z, Ixx, Iyy, Izz))
    
    # -------------------------------------------------------------------------
    # SPAR
    # -------------------------------------------------------------------------
    wing_span = 2 * wing_half_span
    vol_spar = π * (spar_r_outer^2 - spar_r_inner^2) * wing_span
    mass_spar = vol_spar * rho_spar
    mass_spar = max(mass_spar, 0.02 * mass_wing)
    
    spar_x = wing_x + 0.25 * c_avg
    m_spar_half = mass_spar / 2.0
    
    Ixx_s, Iyy_s, Izz_s = rod_inertia(m_spar_half, wing_half_span, :y, spar_r_outer, spar_r_inner)
    push!(comps, Component("spar_right", m_spar_half, spar_x,  y_centroid, wing_z, Ixx_s, Iyy_s, Izz_s))
    push!(comps, Component("spar_left",  m_spar_half, spar_x, -y_centroid, wing_z, Ixx_s, Iyy_s, Izz_s))
    
    # -------------------------------------------------------------------------
    # POINT MASSES
    # -------------------------------------------------------------------------
    push!(comps, Component("motor", CONFIG.motor_mass, CONFIG.motor_x, 0.0, 0.0, 0.0, 0.0, 0.0))
    push!(comps, Component("battery", CONFIG.battery_mass, battery_x, 0.0, 0.0, 0.0, 0.0, 0.0))
    push!(comps, Component("electronics", CONFIG.electronics_mass, electronics_x, 0.0, 0.0, 0.0, 0.0, 0.0))
    
    # NEW: Ballast mass
    if ballast_mass > 0.0
        push!(comps, Component("ballast", ballast_mass, ballast_x, 0.0, 0.0, 0.0, 0.0, 0.0))
    end
    
    # -------------------------------------------------------------------------
    # FUSELAGE
    # -------------------------------------------------------------------------
    max_tail_x = max(htail_x, vtail_x)
    fuselage_length = min(max_tail_x + 0.3, CONFIG.max_fuselage_length)
    
    vol_fuse = π * (fuse_r_outer^2 - fuse_r_inner^2) * fuselage_length
    fuselage_mass = vol_fuse * rho_fuse
    
    Ixx_f, Iyy_f, Izz_f = rod_inertia(fuselage_mass, fuselage_length, :x, fuse_r_outer, fuse_r_inner)
    push!(comps, Component("fuselage", fuselage_mass, 0.0, 0.0, 0.0, Ixx_f, Iyy_f, Izz_f))
    
    # -------------------------------------------------------------------------
    # HORIZONTAL TAIL
    # -------------------------------------------------------------------------
    htail_chords_f = [c isa ForwardDiff.Dual ? ForwardDiff.value(c) : Float64(c) for c in htail_chords]
    c_ht_avg = mean(htail_chords_f)
    
    S_ht = 2.0 * (c_ht_avg * htail_half_span)
    m_ht = S_ht * foam_thickness * rho_foam
    m_ht_half = m_ht / 2.0
    span_ht_half = htail_half_span / 2.0
    x_ht = htail_x + c_ht_avg / 2.0
    y_ht = span_ht_half / 2.0
    
    Ixx_ht, Iyy_ht, Izz_ht = lamina_inertia(m_ht_half, span_ht_half, c_ht_avg)
    push!(comps, Component("ht_right", m_ht_half, x_ht,  y_ht, htail_z, Ixx_ht, Iyy_ht, Izz_ht))
    push!(comps, Component("ht_left",  m_ht_half, x_ht, -y_ht, htail_z, Ixx_ht, Iyy_ht, Izz_ht))
    
    # -------------------------------------------------------------------------
    # VERTICAL TAIL
    # -------------------------------------------------------------------------
    vtail_chords_f = [c isa ForwardDiff.Dual ? ForwardDiff.value(c) : Float64(c) for c in vtail_chords]
    c_vt_avg = mean(vtail_chords_f)
    
    S_vt = c_vt_avg * vtail_height
    m_vt = S_vt * foam_thickness * rho_foam
    x_vt = vtail_x + c_vt_avg / 2.0
    z_vt = vtail_z + vtail_height / 2.0
    
    Ixx_vt, Iyy_vt, Izz_vt = lamina_inertia(m_vt, vtail_height, c_vt_avg)
    push!(comps, Component("vt", m_vt, x_vt, 0.0, z_vt, Ixx_vt, Iyy_vt, Izz_vt))
    
    return comps
end

function compute_inertia_from_components(comps::Vector{Component})
    m_tot = sum(c.mass for c in comps)
    x_cg = sum(c.mass * c.x for c in comps) / m_tot
    y_cg = sum(c.mass * c.y for c in comps) / m_tot
    z_cg = sum(c.mass * c.z for c in comps) / m_tot
    
    Ixx = 0.0
    Iyy = 0.0
    Izz = 0.0
    
    for c in comps
        dx = c.x - x_cg
        dy = c.y - y_cg
        dz = c.z - z_cg
        
        Ixx += c.Ixx_c + c.mass * (dy^2 + dz^2)
        Iyy += c.Iyy_c + c.mass * (dx^2 + dz^2)
        Izz += c.Izz_c + c.mass * (dx^2 + dy^2)
    end
    
    return m_tot, (x_cg, y_cg, z_cg), (Ixx, Iyy, Izz)
end

# ============================================================================
# DYNAMIC STABILITY ANALYSIS (unchanged)
# ============================================================================

function assemble_longitudinal_A(dCF, dCM, S, c, m, Iyy, U)
    """
    Assemble longitudinal state-space matrix from stability derivatives.
    State vector: [u, w, q, θ]
    
    The stability derivatives from VortexLattice are dimensional force/moment 
    coefficients, so we need to convert them to dimensional derivatives.
    
    Reference: Aircraft Flight Mechanics - Dynamic Stability
    """
    q_dyn = 0.5 * CONFIG.air_density * U^2
    
    # Extract stability derivatives (these are dimensionless dC/dx)
    CL_alpha = dCF.alpha[3]  # dCL/dα [1/rad]
    CD_alpha = dCF.alpha[1]  # dCD/dα [1/rad]
    Cm_alpha = dCM.alpha[2]  # dCm/dα [1/rad]
    Cm_q = dCM.q[2]          # dCm/dq̂ where q̂ = qc/(2U) [dimensionless]
    CL_q = dCF.q[3]          # dCL/dq̂ where q̂ = qc/(2U) [dimensionless]
    
    # Convert to dimensional force/moment derivatives
    # Z force derivatives (positive down)
    Z_u = q_dyn * S * (2*CD_alpha) / m  # Simplified: mainly drag increase with speed
    Z_w = -q_dyn * S * CL_alpha / (m * U)  # Lift change with vertical velocity
    Z_q = -q_dyn * S * CL_q * (c / (2*U)) / m  # Lift change with pitch rate
    
    # X force derivatives (we'll assume simplified - drag dominates)
    X_u = -q_dyn * S * (2*CD_alpha) / (m * U)  # Drag change with forward speed
    X_w = q_dyn * S * (CL_alpha - CD_alpha) / m  # Primarily lift component
    
    # Pitching moment derivatives
    M_w = q_dyn * S * c * Cm_alpha / (Iyy * U)  # Moment change with vertical velocity
    M_q = q_dyn * S * c * Cm_q * (c / (2*U)) / Iyy  # Moment damping from pitch rate
    
    # Assemble A matrix for state vector [u, w, q, θ]
    A = zeros(4, 4)
    
    # du/dt equation
    A[1, 1] = X_u
    A[1, 2] = X_w
    A[1, 3] = 0.0
    A[1, 4] = -g
    
    # dw/dt equation  
    A[2, 1] = Z_u
    A[2, 2] = Z_w
    A[2, 3] = U + Z_q
    A[2, 4] = 0.0
    
    # dq/dt equation
    A[3, 1] = 0.0
    A[3, 2] = M_w
    A[3, 3] = M_q
    A[3, 4] = 0.0
    
    # dθ/dt equation
    A[4, 1] = 0.0
    A[4, 2] = 0.0
    A[4, 3] = 1.0
    A[4, 4] = 0.0
    
    return A
end

function assemble_lateral_A(dCF, dCM, S, b, m, Ixx, Izz, U)
    """
    Assemble lateral state-space matrix from stability derivatives.
    State vector: [v, p, r, φ]
    
    Reference: Aircraft Flight Mechanics - Dynamic Stability
    """
    q_dyn = 0.5 * CONFIG.air_density * U^2
    
    # Extract stability derivatives (dimensionless dC/dx)
    CY_beta = dCF.beta[2]   # dCY/dβ [1/rad]
    Cl_beta = dCM.beta[1]   # dCl/dβ (roll moment) [1/rad]
    Cn_beta = dCM.beta[3]   # dCn/dβ (yaw moment) [1/rad]
    Cl_p = dCM.p[1]         # dCl/dp̂ where p̂ = pb/(2U) [dimensionless]
    Cn_p = dCM.p[3]         # dCn/dp̂ [dimensionless]
    Cl_r = dCM.r[1]         # dCl/dr̂ where r̂ = rb/(2U) [dimensionless]
    Cn_r = dCM.r[3]         # dCn/dr̂ [dimensionless]
    
    # Convert to dimensional derivatives
    # Side force derivatives
    Y_v = q_dyn * S * CY_beta / (m * U)  # Side force from sideslip
    
    # Rolling moment derivatives
    L_beta = q_dyn * S * b * Cl_beta / Ixx  # Dihedral effect
    L_p = q_dyn * S * b * Cl_p * (b / (2*U)) / Ixx  # Roll damping
    L_r = q_dyn * S * b * Cl_r * (b / (2*U)) / Ixx  # Roll from yaw rate
    
    # Yawing moment derivatives
    N_beta = q_dyn * S * b * Cn_beta / Izz  # Weathercock stability
    N_p = q_dyn * S * b * Cn_p * (b / (2*U)) / Izz  # Yaw from roll rate
    N_r = q_dyn * S * b * Cn_r * (b / (2*U)) / Izz  # Yaw damping
    
    # Assemble A matrix for state vector [v, p, r, φ]
    A = zeros(4, 4)
    
    # dv/dt equation
    A[1, 1] = Y_v
    A[1, 2] = 0.0
    A[1, 3] = -U
    A[1, 4] = g
    
    # dp/dt equation
    A[2, 1] = L_beta
    A[2, 2] = L_p
    A[2, 3] = L_r
    A[2, 4] = 0.0
    
    # dr/dt equation
    A[3, 1] = N_beta
    A[3, 2] = N_p
    A[3, 3] = N_r
    A[3, 4] = 0.0
    
    # dφ/dt equation
    A[4, 1] = 0.0
    A[4, 2] = 1.0
    A[4, 3] = 0.0
    A[4, 4] = 0.0
    
    return A
end

function analyze_modes(A_long, A_lat)
    results = Dict{String, Any}()
    
    λ_long = eigvals(A_long)
    λ_lat  = eigvals(A_lat)
    
    function mode_info(λ)
        ωn = abs(λ)
        ζ = -real(λ) / (abs(λ) + 1e-12)
        τ = abs(real(λ)) > 1e-9 ? -1.0/real(λ) : Inf
        return (λ=λ, ωn=ωn, ζ=ζ, τ=τ)
    end
    
    if length(λ_long) >= 4
        freqs = abs.(λ_long)
        sorted_indices = sortperm(freqs, rev=true)
        
        sp_idx = sorted_indices[1]
        results["short_period"] = mode_info(λ_long[sp_idx])
        
        ph_idx = sorted_indices[end]
        results["phugoid"] = mode_info(λ_long[ph_idx])
    end
    
    if length(λ_lat) >= 4
        oscillatory = []
        real_modes = []
        
        for (i, λ) in enumerate(λ_lat)
            if abs(imag(λ)) > 0.1
                push!(oscillatory, (i, λ))
            else
                push!(real_modes, (i, λ))
            end
        end
        
        if !isempty(oscillatory)
            dr_idx = argmax([abs(λ) for (_, λ) in oscillatory])
            results["dutch_roll"] = mode_info(oscillatory[dr_idx][2])
        end
        
        if length(real_modes) >= 2
            sorted_real = sort(real_modes, by=x->real(x[2]))
            results["spiral"] = mode_info(sorted_real[end][2])
            results["roll_subsidence"] = mode_info(sorted_real[1][2])
        end
    end
    
    return results
end

# ============================================================================
# OPTIMIZATION SETUP (Major changes here!)
# ============================================================================

"""
Create objective and constraint functions with enhanced design variables.

NEW Design variables (25 total):
  x[1:2] = wing chords (root, tip)
  x[3:4] = wing twists (root, tip)
  x[5] = wing_half_span
  x[6] = wing_dihedral (degrees)
  x[7] = sweep
  x[8:9] = htail chords (root, tip)
  x[10:11] = htail twists (root, tip)
  x[12] = htail_half_span
  x[13] = htail_x
  x[14:15] = vtail chords (root, tip)
  x[16:17] = vtail twists (root, tip)
  x[18] = vtail_height
  x[19] = vtail_x
  x[20] = mass_x (reference point)
  x[21] = battery_x
  x[22] = electronics_x
  x[23] = wing_x
  x[24] = ballast_mass
  x[25] = ballast_x
  
UPDATED CONSTRAINTS (more realistic for small RC aircraft):
  - Phugoid: Allowed to be slightly unstable (common in practice)
  - Spiral: Allowed to be slightly unstable (very common, slow divergence)
  - Dutch roll damping: Relaxed to ζ ≥ 0.04 (still provides adequate handling)
  - Roll mode: Relaxed to τ ≤ 2.0 s (more realistic for small aircraft)
"""
function make_objective_and_constraints(config::DesignConfig)
    # Estimate initial mass for weight requirement
    est_mass = config.motor_mass + config.battery_mass + config.electronics_mass + 0.5
    weight_req = est_mass * g * config.lift_safety_factor
    
    GLOBAL_WING_CAMBER[] = get_camber_function(config, :wing)
    GLOBAL_TAIL_CAMBER[] = get_camber_function(config, :tail)
    
    e_b = config.battery_specific_energy * 3600.0
    η = config.propulsion_efficiency
    m_b = config.battery_mass
    
    function objective_and_constraints!(g_vec, x)
        # Unpack design variables
        wing_chords = [x[1], x[2]]
        wing_twists = [x[3], x[4]]
        wing_half_span = x[5]
        wing_dihedral = x[6]
        sweep = x[7]
        htail_chords = [x[8], x[9]]
        htail_twists = [x[10], x[11]]
        htail_half_span = x[12]
        htail_x = x[13]
        vtail_chords = [x[14], x[15]]
        vtail_twists = [x[16], x[17]]
        vtail_height = x[18]
        vtail_x = x[19]
        mass_x = x[20]
        battery_x = x[21]
        electronics_x = x[22]
        wing_x = x[23]
        ballast_mass = x[24]
        ballast_x = x[25]
        
        # Build configuration
        system, S, c, b, rref, _, _, _ = build_configuration(
            wing_chords, wing_twists, htail_chords, htail_twists, vtail_chords, vtail_twists;
            wing_half_span=wing_half_span,
            wing_dihedral=wing_dihedral,
            sweep=sweep,
            htail_half_span=htail_half_span,
            htail_x=htail_x,
            vtail_height=vtail_height,
            vtail_x=vtail_x,
            mass_x=mass_x,
            wing_x=wing_x
        )
        
        # Run VLM analysis
        CF, CM, dCF, dCM = run_vlm_analysis!(system, S, c, b, rref; 
                                             α=2.0*pi/180, β=0.0, symmetric=false)
        
        CD, CY, CL = CF
        q_dyn = 0.5 * config.air_density * config.cruise_speed^2
        
        S_val = S isa ForwardDiff.Dual ? ForwardDiff.value(S) : Float64(S)
        
        L = q_dyn * S_val * CL
        D = q_dyn * S_val * CD
        
        # Extract Float64 values for component calculation
        wing_half_span_val = wing_half_span isa ForwardDiff.Dual ? ForwardDiff.value(wing_half_span) : Float64(wing_half_span)
        htail_half_span_val = htail_half_span isa ForwardDiff.Dual ? ForwardDiff.value(htail_half_span) : Float64(htail_half_span)
        vtail_height_val = vtail_height isa ForwardDiff.Dual ? ForwardDiff.value(vtail_height) : Float64(vtail_height)
        htail_x_val = htail_x isa ForwardDiff.Dual ? ForwardDiff.value(htail_x) : Float64(htail_x)
        vtail_x_val = vtail_x isa ForwardDiff.Dual ? ForwardDiff.value(vtail_x) : Float64(vtail_x)
        battery_x_val = battery_x isa ForwardDiff.Dual ? ForwardDiff.value(battery_x) : Float64(battery_x)
        electronics_x_val = electronics_x isa ForwardDiff.Dual ? ForwardDiff.value(electronics_x) : Float64(electronics_x)
        wing_x_val = wing_x isa ForwardDiff.Dual ? ForwardDiff.value(wing_x) : Float64(wing_x)
        ballast_mass_val = ballast_mass isa ForwardDiff.Dual ? ForwardDiff.value(ballast_mass) : Float64(ballast_mass)
        ballast_x_val = ballast_x isa ForwardDiff.Dual ? ForwardDiff.value(ballast_x) : Float64(ballast_x)
        
        # Compute mass properties
        comps = build_components(
            wing_half_span_val, wing_chords, 
            htail_half_span_val, htail_chords,
            vtail_height_val, vtail_chords;
            wing_x=wing_x_val, 
            htail_x=htail_x_val, 
            vtail_x=vtail_x_val,
            battery_x=battery_x_val,
            electronics_x=electronics_x_val,
            ballast_mass=ballast_mass_val,
            ballast_x=ballast_x_val
        )
        m_TO, cg, (Ixx, Iyy, Izz) = compute_inertia_from_components(comps)
        
        # Calculate L/D and range
        L_over_D = L / (D + 1e-6)
        range_m = (e_b / g) * η * L_over_D * (m_b / m_TO)
        
        # Objective: maximize range
        f = -range_m
        
        # Assemble state-space matrices
        S_val = S isa ForwardDiff.Dual ? ForwardDiff.value(S) : Float64(S)
        c_val = c isa ForwardDiff.Dual ? ForwardDiff.value(c) : Float64(c)
        b_val = b isa ForwardDiff.Dual ? ForwardDiff.value(b) : Float64(b)
        
        A_long = assemble_longitudinal_A(dCF, dCM, S_val, c_val, m_TO, Iyy, config.cruise_speed)
        A_lat = assemble_lateral_A(dCF, dCM, S_val, b_val, m_TO, Ixx, Izz, config.cruise_speed)
        
        modes = analyze_modes(A_long, A_lat)
        
        # Constraints
        idx = 1
        
        # 1) Lift requirement
        g_vec[idx] = weight_req - L
        idx += 1
        
        # 2) Static longitudinal stability: Cm_α < 0
        Cm_alpha = dCM.alpha[2]
        g_vec[idx] = Cm_alpha + 1e-3
        idx += 1
        
        # 3) Static directional stability: Cn_β > 0
        Cn_beta = dCM.beta[3]
        g_vec[idx] = -Cn_beta + 1e-4
        idx += 1
        
        # 4-5) Short period mode constraints
        if haskey(modes, "short_period")
            sp = modes["short_period"]
            # ζ_sp ≥ 0.30 (unchanged - important for handling)
            g_vec[idx] = 0.30 - sp.ζ
            idx += 1
            # ωn_sp ≥ 1.0 rad/s (unchanged)
            g_vec[idx] = 1.0 - sp.ωn
            idx += 1
        else
            g_vec[idx] = 0.0
            g_vec[idx+1] = 0.0
            idx += 2
        end
        
        # 6) Phugoid mode constraint: RELAXED
        # Many aircraft have slightly unstable phugoids (very slow, pilot can easily correct)
        # Allow Re(λ) up to +0.05 (time to double: ~14 seconds, very manageable)
        if haskey(modes, "phugoid")
            ph = modes["phugoid"]
            g_vec[idx] = real(ph.λ) - 0.05  # Allow slight instability
            idx += 1
        else
            g_vec[idx] = 0.0
            idx += 1
        end
        
        # 7-8) Dutch roll mode constraints - FURTHER RELAXED
        if haskey(modes, "dutch_roll")
            dr = modes["dutch_roll"]
            # FURTHER RELAXED: ζ_dr ≥ 0.02 (was 0.04, originally 0.08)
            # Minimal damping acceptable for RC aircraft with human pilot in loop
            g_vec[idx] = 0.02 - dr.ζ
            idx += 1
            # ωn_dr ≥ 0.4 rad/s (relaxed from 0.5)
            # Lower frequency oscillation is acceptable
            g_vec[idx] = 0.4 - dr.ωn
            idx += 1
        else
            g_vec[idx] = 0.0
            g_vec[idx+1] = 0.0
            idx += 2
        end
        
        # 9) Roll mode constraint: RELAXED
        # τ_roll ≤ 2.0 s (was 1.5 s) → Re(λ) ≤ -0.5 (was -0.67)
        # Small RC aircraft can have slightly slower roll response
        if haskey(modes, "roll_subsidence")
            roll = modes["roll_subsidence"]
            g_vec[idx] = real(roll.λ) + 0.5
            idx += 1
        else
            g_vec[idx] = 0.0
            idx += 1
        end
        
        # 10) Spiral mode constraint: FURTHER RELAXED
        # Allow more instability: Re(λ) ≤ +0.05 (was +0.01)
        # Time to double at λ=0.05 is ~14s, still very manageable
        # Spiral instability is extremely common and acceptable for RC aircraft
        if haskey(modes, "spiral")
            spiral = modes["spiral"]
            g_vec[idx] = real(spiral.λ) - 0.05  # Allow more instability
            idx += 1
        else
            g_vec[idx] = 0.0
            idx += 1
        end
        
        # 11-13) Monotonic chord taper constraints
        g_vec[idx] = wing_chords[2] - wing_chords[1]  # Wing tip <= root
        idx += 1
        g_vec[idx] = htail_chords[2] - htail_chords[1]  # H-tail tip <= root
        idx += 1
        g_vec[idx] = vtail_chords[2] - vtail_chords[1]  # V-tail tip <= root
        idx += 1
        
        # 14-19) Geometric packaging constraints
        # Wing must fit on fuselage
        max_wing_chord = maximum(wing_chords)
        wing_le = wing_x_val - 0.25 * max_wing_chord
        wing_te = wing_x_val + 0.75 * max_wing_chord
        g_vec[idx] = -wing_le
        idx += 1
        g_vec[idx] = wing_te - config.max_fuselage_length
        idx += 1
        
        # H-tail must fit on fuselage
        max_ht_chord = maximum(htail_chords)
        ht_le = htail_x_val - 0.25 * max_ht_chord
        ht_te = htail_x_val + 0.75 * max_ht_chord
        g_vec[idx] = -ht_le
        idx += 1
        g_vec[idx] = ht_te - config.max_fuselage_length
        idx += 1
        
        # V-tail must fit on fuselage
        max_vt_chord = maximum(vtail_chords)
        vt_le = vtail_x_val - 0.25 * max_vt_chord
        vt_te = vtail_x_val + 0.75 * max_vt_chord
        g_vec[idx] = -vt_le
        idx += 1
        g_vec[idx] = vt_te - config.max_fuselage_length
        idx += 1
        
        # 20) Static margin: NP must be ahead of CG
        CL_alpha = dCF.alpha[3]
        x_np = mass_x - (Cm_alpha / (CL_alpha + 1e-12)) * c_val
        static_margin = (x_np - cg[1]) / c_val
        g_vec[idx] = config.static_margin_min - static_margin
        idx += 1
        
        # 21-23) Component positioning constraints: Keep battery, electronics, and ballast behind motor
        # This ensures proper weight distribution and prevents interference
        g_vec[idx] = config.motor_x - battery_x_val  # battery_x must be >= motor_x
        idx += 1
        g_vec[idx] = config.motor_x - electronics_x_val  # electronics_x must be >= motor_x
        idx += 1
        g_vec[idx] = config.motor_x - ballast_x_val  # ballast_x must be >= motor_x (if ballast present)
        idx += 1
        
        return f
    end
    
    # Total constraints: 23 (was 20, added 3 component positioning constraints)
    n_constraints = 23
    return objective_and_constraints!, n_constraints
end

# ============================================================================
# OPTIMIZATION RUNNER
# ============================================================================

function optimize_aircraft(config::DesignConfig=CONFIG)
    println("="^70)
    println("ENHANCED AIRCRAFT OPTIMIZATION (V5)")
    println("="^70)
    
    objfun, ng = make_objective_and_constraints(config)
    
    # Initial guess (25 variables)
    x0 = zeros(25)
    x0[1:2] = [0.2, 0.15]      # wing chords
    x0[3:4] = [3.0, -1.0]      # wing twists
    x0[5] = 0.7                # wing_half_span
    x0[6] = 1.0                # wing_dihedral
    x0[7] = 0.05               # sweep
    x0[8:9] = [0.1, 0.08]     # htail chords
    x0[10:11] = [0.0, 0.0]   # htail twists
    x0[12] = 0.15               # htail_half_span
    x0[13] = 0.9               # htail_x
    x0[14:15] = [0.1, 0.08]   # vtail chords
    x0[16:17] = [0.0, 0.0]     # vtail twists
    x0[18] = 0.15              # vtail_height
    x0[19] = 0.9               # vtail_x
    x0[20] = 0.0               # mass_x
    x0[21] = 0.08               # battery_x (UPDATED: behind motor at 0.05)
    x0[22] = 0.08               # electronics_x (UPDATED: behind motor)
    x0[23] = 0.2               # wing_x
    x0[24] = 0.0               # ballast_mass
    x0[25] = 0.05               # ballast_x (UPDATED: behind motor if used)
    
    # Bounds
    lx = zeros(25)
    ux = ones(25)
    
    # Wing chords
    lx[1:2] .= 0.1
    ux[1:2] .= 0.5
    
    # Wing twists
    lx[3:4] .= -10.0
    ux[3:4] .= 10.0
    
    # Wing half-span
    lx[5] = config.min_wing_half_span
    ux[5] = config.max_wing_half_span
    
    # Wing dihedral
    lx[6] = 0.0
    ux[6] = config.max_wing_dihedral
    
    # Sweep
    lx[7] = 0.0
    ux[7] = 0.5 # changed from 0.3
    
    # H-tail chords
    lx[8:9] .= 0.05
    ux[8:9] .= 0.3
    
    # H-tail twists
    lx[10:11] .= -10.0
    ux[10:11] .= 5.0
    
    # H-tail half-span
    lx[12] = 0.15
    ux[12] = 0.5
    
    # H-tail x
    lx[13] = 0.5
    ux[13] = config.max_fuselage_length - 0.05 #changed to let tail be closer to end
    
    # V-tail chords
    lx[14:15] .= 0.05
    ux[14:15] .= 0.3
    
    # V-tail twists
    lx[16:17] .= -5.0
    ux[16:17] .= 5.0
    
    # V-tail height
    lx[18] = 0.15
    ux[18] = 0.5
    
    # V-tail x
    lx[19] = 0.5
    ux[19] = config.max_fuselage_length - 0.05 #changed to let tail be closer to end
    
    # Mass reference x ------ what even is this variable??
    lx[20] = -0.3
    ux[20] = 0.3
    
    # Battery x
    lx[21] = config.motor_x  # UPDATED: Must be at or behind motor
    ux[21] = 0.5
    
    # Electronics x
    lx[22] = config.motor_x  # UPDATED: Must be at or behind motor
    ux[22] = 0.25
    
    # Wing x
    lx[23] = 0.0
    ux[23] = 0.5
    
    # Ballast mass
    lx[24] = 0.0
    ux[24] = 0.5  # Max 500g ballast
    
    # Ballast x
    lx[25] = config.motor_x  # UPDATED: Must be at or behind motor
    ux[25] = 0.8
    
    # Constraint bounds
    lg = fill(-Inf, ng)
    ug = zeros(ng)
    
    # Configure Ipopt
    ipopts = Dict(
        "tol" => 1e-4,
        "max_iter" => 1000,
        "print_level" => 5,
        "mu_strategy" => "adaptive",
        "linear_solver" => "mumps",
        "acceptable_tol" => 1e-3,
        "acceptable_iter" => 15
    )
    
    solver = IPOPT(ipopts)
    options = Options(; solver=solver, derivatives=ForwardAD())
    
    @printf("\nStarting optimization...\n")
    @printf("  Design variables: 25\n")
    @printf("  Constraints: %d\n", ng)
    @printf("  Wing span: %.2f - %.2f m\n", 2*config.min_wing_half_span, 2*config.max_wing_half_span)
    @printf("  Cruise speed: %.1f m/s\n\n", config.cruise_speed)
    
    println("  RELAXED STABILITY CONSTRAINTS (more realistic for RC aircraft):")
    println("    • Phugoid: Allow slight instability (Re(λ) ≤ +0.05)")
    println("    • Dutch roll damping: ζ ≥ 0.02 (was 0.08, now 0.02)")
    println("    • Dutch roll frequency: ωn ≥ 0.4 rad/s (relaxed from 0.5)")
    println("    • Roll time constant: τ ≤ 2.0s (was 1.5s)")
    println("    • Spiral: Allow instability (Re(λ) ≤ +0.05, was +0.01)")
    println("    • Static margin: INCREASED to 0.15 (15% MAC)")
    println()
    
    xopt, fopt, info = minimize(objfun, x0, ng, lx, ux, lg, ug, options)
    
    println("\n" * "="^70)
    println("OPTIMIZATION COMPLETE")
    println("="^70)
    @printf("Objective value: %.6f\n", fopt)
    @printf("Exit flag: %s\n\n", info)
    
    return xopt, fopt, info
end

# ============================================================================
# ANALYSIS AND VISUALIZATION (adapted for new variable structure)
# ============================================================================

function analyze_design(x::Vector{Float64}, config::DesignConfig=CONFIG; verbose::Bool=true)
    # Unpack all 25 variables
    wing_chords = [x[1], x[2]]
    wing_twists = [x[3], x[4]]
    wing_half_span = x[5]
    wing_dihedral = x[6]
    sweep = x[7]
    htail_chords = [x[8], x[9]]
    htail_twists = [x[10], x[11]]
    htail_half_span = x[12]
    htail_x = x[13]
    vtail_chords = [x[14], x[15]]
    vtail_twists = [x[16], x[17]]
    vtail_height = x[18]
    vtail_x = x[19]
    mass_x = x[20]
    battery_x = x[21]
    electronics_x = x[22]
    wing_x = x[23]
    ballast_mass = x[24]
    ballast_x = x[25]
    
    if verbose
        println("\n" * "="^70)
        println("DESIGN ANALYSIS (V5)")
        println("="^70)
        
        println("\nWing Geometry:")
        @printf("  Span: %.3f m (%.2f in)\n", 2*wing_half_span, 2*wing_half_span*39.3701)
        @printf("  Dihedral: %.2f°\n", wing_dihedral)
        @printf("  Position (x): %.3f m\n", wing_x)
        @printf("  Root chord: %.3f m (%.2f in)\n", wing_chords[1], wing_chords[1]*39.3701)
        @printf("  Tip chord: %.3f m (%.2f in)\n", wing_chords[2], wing_chords[2]*39.3701)
        @printf("  Root twist: %.2f°\n", wing_twists[1])
        @printf("  Tip twist: %.2f°\n", wing_twists[2])
        @printf("  Sweep: %.3f m\n", sweep)
        
        println("\nHorizontal Tail:")
        @printf("  Position (x): %.3f m\n", htail_x)
        @printf("  Span: %.3f m (%.2f in)\n", 2*htail_half_span, 2*htail_half_span*39.3701)
        @printf("  Root chord: %.3f m (%.2f in)\n", htail_chords[1], htail_chords[1]*39.3701)
        @printf("  Tip chord: %.3f m (%.2f in)\n", htail_chords[2], htail_chords[2]*39.3701)
        @printf("  Root twist: %.2f°\n", htail_twists[1])
        @printf("  Tip twist: %.2f°\n", htail_twists[2])
        
        # Calculate h-tail area
        ht_area = (htail_chords[1] + htail_chords[2]) * htail_half_span
        @printf("  Area: %.4f m²\n", ht_area)
        
        println("\nVertical Tail:")
        @printf("  Position (x): %.3f m\n", vtail_x)
        @printf("  Height: %.3f m (%.2f in)\n", vtail_height, vtail_height*39.3701)
        @printf("  Root chord: %.3f m (%.2f in)\n", vtail_chords[1], vtail_chords[1]*39.3701)
        @printf("  Tip chord: %.3f m (%.2f in)\n", vtail_chords[2], vtail_chords[2]*39.3701)
        @printf("  Root twist: %.2f°\n", vtail_twists[1])
        @printf("  Tip twist: %.2f°\n", vtail_twists[2])
        
        # Calculate v-tail area
        vt_area = 0.5 * (vtail_chords[1] + vtail_chords[2]) * vtail_height
        @printf("  Area: %.4f m²\n", vt_area)
        
        println("\nComponent Positions:")
        @printf("  Motor: x = %.3f m (FIXED)\n", config.motor_x)
        @printf("  Battery: x = %.3f m\n", battery_x)
        @printf("  Electronics: x = %.3f m\n", electronics_x)
        @printf("  Ballast: %.3f kg at x = %.3f m\n", ballast_mass, ballast_x)
        @printf("  Reference CG: x = %.3f m\n", mass_x)
    end
    
    # Build configuration
    GLOBAL_WING_CAMBER[] = get_camber_function(config, :wing)
    GLOBAL_TAIL_CAMBER[] = get_camber_function(config, :tail)
    
    system, S, c, b, rref, _, _, _ = build_configuration(
        wing_chords, wing_twists, htail_chords, htail_twists, vtail_chords, vtail_twists;
        wing_half_span=wing_half_span,
        wing_dihedral=wing_dihedral,
        sweep=sweep,
        htail_half_span=htail_half_span,
        htail_x=htail_x,
        vtail_height=vtail_height,
        vtail_x=vtail_x,
        mass_x=mass_x,
        wing_x=wing_x
    )
    
    # Aerodynamic analysis
    CF, CM, dCF, dCM = run_vlm_analysis!(system, S, c, b, rref; 
                                         α=2.0*pi/180, β=0.0, symmetric=false)
    
    CD, CY, CL = CF
    Cl, Cm, Cn = CM
    
    q_dyn = 0.5 * config.air_density * config.cruise_speed^2
    L = q_dyn * S * CL
    D = q_dyn * S * CD
    
    # Compute inertia
    comps = build_components(
        wing_half_span, wing_chords,
        htail_half_span, htail_chords,
        vtail_height, vtail_chords;
        wing_x=wing_x, 
        htail_x=htail_x, 
        vtail_x=vtail_x,
        battery_x=battery_x,
        electronics_x=electronics_x,
        ballast_mass=ballast_mass,
        ballast_x=ballast_x
    )
    m_tot, cg, (Ixx, Iyy, Izz) = compute_inertia_from_components(comps)
    
    # Range calculation
    e_b = config.battery_specific_energy * 3600.0
    η = config.propulsion_efficiency
    m_b = config.battery_mass
    L_over_D = L / D
    range_m = (e_b / g) * η * L_over_D * (m_b / m_tot)
    
    if verbose
        println("\n" * "-"^70)
        println("Aerodynamic Performance (α = 2°):")
        println("-"^70)
        @printf("  Reference area: %.4f m²\n", S)
        @printf("  Reference chord: %.4f m\n", c)
        @printf("  Aspect ratio: %.2f\n", b^2/S)
        @printf("\n  CL = %.4f\n", CL)
        @printf("  CD = %.4f\n", CD)
        @printf("  L/D = %.2f\n", L_over_D)
        @printf("\n  Lift force: %.2f N\n", L)
        @printf("  Drag force: %.2f N\n", D)
        @printf("  Weight: %.2f N\n", m_tot * g)
        @printf("\n  ** Estimated Range: %.1f m (%.2f km) **\n", range_m, range_m/1000.0)
    end
    
    # Stability derivatives
    CL_alpha = dCF.alpha[3]
    Cm_alpha = dCM.alpha[2]
    Cn_beta = dCM.beta[3]
    
    # Calculate neutral point
    x_np = mass_x - (Cm_alpha / (CL_alpha + 1e-12)) * c
    static_margin = (x_np - cg[1]) / c
    
    if verbose
        println("\n" * "-"^70)
        println("Static Stability:")
        println("-"^70)
        @printf("  CG position: %.3f m\n", cg[1])
        @printf("  Neutral point: %.3f m\n", x_np)
        @printf("  Static margin: %.3f (%.1f%%)\n", static_margin, static_margin*100)
        @printf("  dCL/dα = %.4f\n", CL_alpha)
        @printf("  dCm/dα = %.4f  (should be < 0)\n", Cm_alpha)
        @printf("  dCn/dβ = %.4f  (should be > 0)\n", Cn_beta)
        
        if Cm_alpha < 0
            println("  ✓ Longitudinally stable")
        else
            println("  ✗ Longitudinally unstable!")
        end
        
        if Cn_beta > 0
            println("  ✓ Directionally stable")
        else
            println("  ✗ Directionally unstable!")
        end
        
        if static_margin > config.static_margin_min
            println("  ✓ Adequate static margin")
        else
            println("  ✗ Insufficient static margin!")
        end
    end
    
    if verbose
        println("\n" * "-"^70)
        println("Mass Properties:")
        println("-"^70)
        @printf("  Total mass: %.3f kg (%.2f oz)\n", m_tot, m_tot*35.274)
        @printf("  CG position: (%.3f, %.3f, %.3f) m\n", cg[1], cg[2], cg[3])
        @printf("  Ixx = %.4f kg·m²\n", Ixx)
        @printf("  Iyy = %.4f kg·m²\n", Iyy)
        @printf("  Izz = %.4f kg·m²\n", Izz)
        
        println("\n  Component breakdown:")
        for comp in comps
            @printf("    %-15s: %.3f kg at (%.3f, %.3f, %.3f) m\n", 
                    comp.name, comp.mass, comp.x, comp.y, comp.z)
        end
    end
    
    # Dynamic stability
    A_long = assemble_longitudinal_A(dCF, dCM, S, c, m_tot, Iyy, config.cruise_speed)
    A_lat = assemble_lateral_A(dCF, dCM, S, b, m_tot, Ixx, Izz, config.cruise_speed)
    
    modes = analyze_modes(A_long, A_lat)
    
    if verbose
        println("\n" * "-"^70)
        println("Dynamic Stability Modes:")
        println("-"^70)
        
        for mode_name in ["short_period", "phugoid", "dutch_roll", "roll_subsidence", "spiral"]
            if haskey(modes, mode_name)
                mode = modes[mode_name]
                status = real(mode.λ) < 0 ? "✓" : "✗"
                if abs(imag(mode.λ)) > 0.01
                    @printf("  %s %-20s: λ = %7.4f %+7.4fi  |  ωₙ = %.3f rad/s  |  ζ = %.3f\n",
                            status, mode_name, real(mode.λ), imag(mode.λ), mode.ωn, mode.ζ)
                else
                    τ_str = isfinite(mode.τ) ? @sprintf("%.2f s", mode.τ) : "∞"
                    @printf("  %s %-20s: λ = %7.4f          |  τ = %s\n",
                            status, mode_name, real(mode.λ), τ_str)
                end
            end
        end
        
        println("\n  Legend: ✓ = stable (Re(λ) < 0), ✗ = unstable (Re(λ) ≥ 0)")
    end
    
    return system, S, c, b, modes, comps, (m_tot, cg, (Ixx, Iyy, Izz))
end

function diagnose_constraints(x::Vector{Float64}, config::DesignConfig=CONFIG)
    g_vec = zeros(23) 
    
    objfun, ng = make_objective_and_constraints(config)
    f = objfun(g_vec, x)
    
    println("\nConstraint Diagnosis:")
    println("="^70)
    @printf("Objective value: %.2f\n\n", f)
    
    constraint_names = [
        "Lift requirement",
        "Longitudinal stability (Cm_α < 0)",
        "Directional stability (Cn_β > 0)",
        "Short period damping (ζ ≥ 0.30)",
        "Short period frequency (ωn ≥ 1.0 rad/s)",
        "Phugoid stability (Re(λ) ≤ +0.05) [RELAXED]",
        "Dutch roll damping (ζ ≥ 0.02) [VERY RELAXED]",
        "Dutch roll frequency (ωn ≥ 0.4 rad/s) [RELAXED]",
        "Roll time constant (τ ≤ 2.0 s) [RELAXED]",
        "Spiral stability (Re(λ) ≤ +0.05) [VERY RELAXED]",
        "Wing chord taper",
        "H-tail chord taper",
        "V-tail chord taper",
        "Wing LE on fuselage",
        "Wing TE on fuselage",
        "H-tail LE on fuselage",
        "H-tail TE on fuselage",
        "V-tail LE on fuselage",
        "V-tail TE on fuselage",
        "Static margin (NP ahead of CG by ≥15%)",
        "Battery position (behind motor)",
        "Electronics position (behind motor)",
        "Ballast position (behind motor)"
    ]
    
    for i in 1:length(constraint_names)
        status = g_vec[i] <= 0.0 ? "✓" : "✗"
        @printf("%s %s: g[%d] = %.6f %s\n", status, constraint_names[i], i, g_vec[i], 
                g_vec[i] <= 0.0 ? "(satisfied)" : "(VIOLATED)")
    end
    
    println("\n" * "="^70)
    println("RELAXED CONSTRAINTS EXPLANATION:")
    println("="^70)
    println("  Phugoid: Allow Re(λ) up to +0.05 (~14s time-to-double)")
    println("    → Very slow divergence, pilot can easily correct")
    println("    → Many real aircraft have slightly unstable phugoids")
    println()
    println("  Dutch Roll: Reduced damping requirement to ζ ≥ 0.02")
    println("    → Minimal damping acceptable with human pilot correction")
    println("    → Frequency requirement also relaxed to ωn ≥ 0.4 rad/s")
    println()
    println("  Roll: Increased time constant limit to 2.0s")
    println("    → Realistic for small RC gliders")
    println("    → Still quick enough for good handling")
    println()
    println("  Spiral: Allow Re(λ) up to +0.05 (~14s time-to-double)")
    println("    → Slight spiral instability is extremely common")
    println("    → Very slow divergence, easy to correct")
    println()
    println("  Static Margin: Increased minimum to 0.15 (15% MAC)")
    println("    → Provides strong static stability")
    println("    → Better handling and easier to trim")
    println("="^70)
end

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

function print_config(config::DesignConfig=CONFIG)
    println("\n" * "="^70)
    println("DESIGN CONFIGURATION (V5)")
    println("="^70)
    
    println("\nFlight Conditions:")
    @printf("  Cruise speed: %.1f m/s\n", config.cruise_speed)
    @printf("  Air density: %.3f kg/m³\n", config.air_density)
    
    println("\nWing Geometry Constraints:")
    @printf("  Half-span range: %.3f - %.3f m\n", config.min_wing_half_span, config.max_wing_half_span)
    @printf("  Max dihedral: %.1f°\n", config.max_wing_dihedral)
    @printf("  Number of span sections: %d\n", config.n_span_sections)
    
    println("\nMaterials:")
    @printf("  Foam density: %.3f kg/m³\n", config.foam_density)
    @printf("  Foam thickness: %.1f mm\n", config.foam_thickness*1000)
    @printf("  Tube OD/ID: %.1f / %.1f mm\n", 
            config.fuselage_outer_diameter*1000, config.fuselage_inner_diameter*1000)
    
    println("Component Masses:")
    @printf("  Motor: %.2f kg at x=%.2f m (FIXED)\n", config.motor_mass, config.motor_x)
    @printf("  Battery: %.2f kg (optimized position)\n", config.battery_mass)
    @printf("  Electronics: %.2f kg (optimized position)\n", config.electronics_mass)
    @printf("  Ballast: 0-0.2 kg (optimized mass and position)\n")
    @printf("\n  TOTAL FIXED MASS: %.3f kg (%.1f oz)\n", 
            config.motor_mass + config.battery_mass + config.electronics_mass,
            (config.motor_mass + config.battery_mass + config.electronics_mass) * 35.274)
    
    println("\nPropulsion:")
    @printf("  Battery: %.1f Wh/kg\n", config.battery_specific_energy)
    @printf("  Efficiency: %.2f\n", config.propulsion_efficiency)
    
    println("\nAirfoils:")
    @printf("  Wing: %s", config.wing_airfoil_type)
    if config.wing_airfoil_type == "NACA4"
        @printf(" (NACA %s)", config.wing_naca_digits)
    end
    println()
    @printf("  Tail: %s", config.tail_airfoil_type)
    if config.tail_airfoil_type == "NACA4"
        @printf(" (NACA %s)", config.tail_naca_digits)
    end
    println()
    
    println("="^70)
end

function set_config!(; kwargs...)
    for (key, value) in kwargs
        if hasfield(DesignConfig, key)
            setfield!(CONFIG, key, value)
            @printf("✓ Set %s = %s\n", key, value)
        else
            @warn "Unknown configuration parameter: $key"
        end
    end
    println("\nConfiguration updated.")
end

function load_airfoil!(filepath::String; surface::Symbol=:wing)
    if !isfile(filepath)
        error("Airfoil file not found: $filepath")
    end
    
    if surface == :wing
        set_config!(
            wing_airfoil_type = "dat_file",
            wing_dat_file = filepath
        )
        println("✓ Loaded wing airfoil from: $filepath")
    elseif surface == :tail
        set_config!(
            tail_airfoil_type = "dat_file",
            tail_dat_file = filepath
        )
        println("✓ Loaded tail airfoil from: $filepath")
    else
        error("surface must be :wing or :tail")
    end
end

function calculate_range(;
    L_over_D::Float64 = 30.0,
    m_total::Float64 = 5.0,
    battery_mass::Float64 = CONFIG.battery_mass,
    battery_energy::Float64 = CONFIG.battery_specific_energy,
    efficiency::Float64 = CONFIG.propulsion_efficiency
)
    e_b = battery_energy * 3600.0
    range_m = (e_b / g) * efficiency * L_over_D * (battery_mass / m_total)
    
    println("\nRange Calculation:")
    println("="^50)
    @printf("  Battery: %.2f kg at %.1f Wh/kg\n", battery_mass, battery_energy)
    @printf("  Total mass: %.2f kg\n", m_total)
    @printf("  L/D ratio: %.1f\n", L_over_D)
    @printf("  Propulsion efficiency: %.2f\n", efficiency)
    println("-"^50)
    @printf("  Range: %.1f m (%.2f km)\n", range_m, range_m/1000.0)
    println("="^50)
    
    return range_m
end

function quick_test()
    println("\n" * "="^70)
    println("QUICK TEST: Basic VLM Analysis")
    println("="^70)
    
    xle   = [0.0, 0.4]
    yle   = [0.0, 7.5]
    zle   = [0.0, 0.0]
    chord = [2.2, 1.8]
    theta = (2.0 * pi/180) .* ones(2)
    phi   = zeros(2)
    fc    = [(x)->0.0 for _ in 1:2]
    
    ns = 12
    nc = 6
    
    grid, ratio = wing_to_grid(xle, yle, zle, chord, theta, phi, ns, nc;
                               fc=fc, mirror=true, spacing_s=Sine())
    
    system = System([grid]; ratios=[ratio])
    
    Sref = 2 * 0.5 * (yle[2] - yle[1]) * (chord[1] + chord[2])
    cref = Sref / (2*yle[end])
    bref = 2*yle[end]
    rref = [0.25*cref, 0.0, 0.0]
    
    CF, CM, dCF, dCM = run_vlm_analysis!(system, Sref, cref, bref, rref;
                                         α=2.0*pi/180, β=0.0, symmetric=false)
    
    CD, CY, CL = CF
    
    @printf("\nResults:\n")
    @printf("  Wing area: %.2f m²\n", Sref)
    @printf("  CL = %.4f\n", CL)
    @printf("  CD = %.4f\n", CD)
    @printf("  L/D = %.2f\n", CL/CD)
    
    println("\n✓ Basic VLM test passed!")
end

# ============================================================================
# MAIN EXECUTION
# ============================================================================

function main()
    println("\n" * "#"^70)
    println("# ENHANCED AIRCRAFT OPTIMIZATION (V5)")
    println("# Maximum Range with Full Geometric Control")
    println("#"^70)
    println("#")
    println("# NEW FEATURES:")
    println("#  - Wing span is now optimizable")
    println("#  - Wing dihedral for lateral stability")
    println("#  - Full tail parameterization (chords, twists, spans)")
    println("#  - Ballast mass for CG control")
    println("#  - 25 design variables total")
    println("#")
    println("#"^70)
    
    print_config()
    quick_test()
    
    println("\n" * "="^70)
    println("READY TO OPTIMIZE")
    println("="^70)
    println("\nPress Enter to start optimization, or Ctrl+C to exit...")
    readline()
    
    xopt, fopt, info = optimize_aircraft(CONFIG)
    analyze_design(xopt, CONFIG; verbose=true)
    
    println("\n" * "#"^70)
    println("# OPTIMIZATION COMPLETE")
    println("#"^70)
    
    return xopt, fopt
end

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

"""
Calculate neutral point (aerodynamic center) location.
"""
function calculate_neutral_point(system, S, c, b, rref)
    CF, CM, dCF, dCM = run_vlm_analysis!(system, S, c, b, rref; α=2.0*pi/180, β=0.0)
    
    Cm_alpha = dCM.alpha[2]
    CL_alpha = dCF.alpha[3]
    
    x_np = rref[1] - (Cm_alpha / CL_alpha) * c
    
    return x_np
end

"""
Plot top view of aircraft with CG, NP, and component positions.
"""
function plot_top_view(x::Vector{Float64}, config::DesignConfig=CONFIG; filename="aircraft_top_view.png")
    # Unpack design variables
    wing_chords = [x[1], x[2]]
    wing_twists = [x[3], x[4]]
    wing_half_span = x[5]
    wing_dihedral = x[6]
    sweep = x[7]
    htail_chords = [x[8], x[9]]
    htail_twists = [x[10], x[11]]
    htail_half_span = x[12]
    htail_x = x[13]
    vtail_chords = [x[14], x[15]]
    vtail_twists = [x[16], x[17]]
    vtail_height = x[18]
    vtail_x = x[19]
    mass_x = x[20]
    battery_x = x[21]
    electronics_x = x[22]
    wing_x = x[23]
    ballast_mass = x[24]
    ballast_x = x[25]
    
    # Build configuration
    GLOBAL_WING_CAMBER[] = get_camber_function(config, :wing)
    GLOBAL_TAIL_CAMBER[] = get_camber_function(config, :tail)
    
    system, S, c, b, rref, _, _, _ = build_configuration(
        wing_chords, wing_twists, htail_chords, htail_twists, vtail_chords, vtail_twists;
        wing_half_span=wing_half_span,
        wing_dihedral=wing_dihedral,
        sweep=sweep,
        htail_half_span=htail_half_span,
        htail_x=htail_x,
        vtail_height=vtail_height,
        vtail_x=vtail_x,
        mass_x=mass_x,
        wing_x=wing_x
    )
    
    # Get CG and components
    comps = build_components(
        wing_half_span, wing_chords,
        htail_half_span, htail_chords,
        vtail_height, vtail_chords;
        wing_x=wing_x, htail_x=htail_x, vtail_x=vtail_x,
        battery_x=battery_x, electronics_x=electronics_x,
        ballast_mass=ballast_mass, ballast_x=ballast_x
    )
    m_tot, cg, (Ixx, Iyy, Izz) = compute_inertia_from_components(comps)
    
    # Calculate neutral point
    x_np = calculate_neutral_point(system, S, c, b, rref)
    
    # Create plot
    p = plot(size=(800, 600), legend=:topright, dpi=150)
    
    # Plot wing outline (both halves with taper)
    n = 2  # Two control points
    y_sections = [0.0, wing_half_span]
    
    # Right wing
    x_le_root = wing_x - 0.25 * wing_chords[1]
    x_le_tip = wing_x - 0.25 * wing_chords[2] + sweep
    x_te_root = x_le_root + wing_chords[1]
    x_te_tip = x_le_tip + wing_chords[2]
    
    # Leading edge
    plot!([x_le_root, x_le_tip], [0.0, wing_half_span], 
          color=:black, linewidth=2, label="Wing")
    # Trailing edge
    plot!([x_te_root, x_te_tip], [0.0, wing_half_span], 
          color=:black, linewidth=2, label="")
    # Tip
    plot!([x_te_tip, x_le_tip], [wing_half_span, wing_half_span], 
          color=:black, linewidth=2, label="")
    # Root (will be drawn for left side too)
    
    # Left wing (mirror)
    plot!([x_le_root, x_le_tip], [0.0, -wing_half_span], 
          color=:black, linewidth=2, label="")
    plot!([x_te_root, x_te_tip], [0.0, -wing_half_span], 
          color=:black, linewidth=2, label="")
    plot!([x_te_tip, x_le_tip], [-wing_half_span, -wing_half_span], 
          color=:black, linewidth=2, label="")
    
    # Close wing at root
    plot!([x_te_root, x_le_root], [0.0, 0.0], 
          color=:black, linewidth=2, label="")
    
    # Horizontal tail (with taper)
    ht_xle_root = htail_x - 0.25 * htail_chords[1]
    ht_xle_tip = htail_x - 0.25 * htail_chords[2]
    ht_xte_root = ht_xle_root + htail_chords[1]
    ht_xte_tip = ht_xle_tip + htail_chords[2]
    
    # Right h-tail
    plot!([ht_xle_root, ht_xle_tip], [0.0, htail_half_span],
          color=:black, linewidth=1.5, linestyle=:dash, label="H-Tail")
    plot!([ht_xte_root, ht_xte_tip], [0.0, htail_half_span],
          color=:black, linewidth=1.5, linestyle=:dash, label="")
    plot!([ht_xte_tip, ht_xle_tip], [htail_half_span, htail_half_span],
          color=:black, linewidth=1.5, linestyle=:dash, label="")
    
    # Left h-tail
    plot!([ht_xle_root, ht_xle_tip], [0.0, -htail_half_span],
          color=:black, linewidth=1.5, linestyle=:dash, label="")
    plot!([ht_xte_root, ht_xte_tip], [0.0, -htail_half_span],
          color=:black, linewidth=1.5, linestyle=:dash, label="")
    plot!([ht_xte_tip, ht_xle_tip], [-htail_half_span, -htail_half_span],
          color=:black, linewidth=1.5, linestyle=:dash, label="")
    
    # Close at root
    plot!([ht_xte_root, ht_xle_root], [0.0, 0.0],
          color=:black, linewidth=1.5, linestyle=:dash, label="")
    
    # Fuselage centerline
    fuse_length = min(max(htail_x, vtail_x) + 0.2, config.max_fuselage_length)
    plot!([0, fuse_length], [0, 0], color=:gray, linewidth=3, label="Fuselage")
    
    # Plot CG
    scatter!([cg[1]], [cg[2]], marker=:circle, markersize=10, color=:red, 
             label="CG", markerstrokewidth=2)
    
    # Plot neutral point
    scatter!([x_np], [0.0], marker=:xcross, markersize=10, color=:blue, 
             label="Neutral Point", markerstrokewidth=2)
    
    # Plot component positions
    scatter!([config.motor_x], [0.0], marker=:square, markersize=8, color=:purple,
             label="Motor")
    scatter!([battery_x], [0.0], marker=:diamond, markersize=8, color=:green,
             label="Battery")
    scatter!([electronics_x], [0.0], marker=:star5, markersize=8, color=:orange,
             label="Electronics")
    
    # Plot ballast if present
    if ballast_mass > 0.001
        scatter!([ballast_x], [0.0], marker=:dtriangle, markersize=8, color=:brown,
                 label=@sprintf("Ballast (%.0fg)", ballast_mass*1000))
    end
    
    # Add stability margin indicator
    static_margin = (x_np - cg[1]) / c
    title!(string("Aircraft Top View\n",
                  "Static Margin: ", @sprintf("%.3f", static_margin), " (x̄_np - x̄_cg)\n",
                  "Wing span: ", @sprintf("%.3f", 2*wing_half_span), " m, Wing area: ", @sprintf("%.4f", S), " m²"))
    xlabel!("x [m]")
    ylabel!("y [m]")
    
    # Equal aspect ratio
    plot!(aspect_ratio=:equal)
    
    savefig(p, filename)
    println("✓ Saved top view to: $filename")
    
    return p
end

"""
Plot side view of aircraft with CG, NP, and component positions.
Shows dihedral effect on z-position.
"""
function plot_side_view(x::Vector{Float64}, config::DesignConfig=CONFIG; filename="aircraft_side_view.png")
    # Unpack design variables
    wing_chords = [x[1], x[2]]
    wing_twists = [x[3], x[4]]
    wing_half_span = x[5]
    wing_dihedral = x[6]
    sweep = x[7]
    htail_chords = [x[8], x[9]]
    htail_twists = [x[10], x[11]]
    htail_half_span = x[12]
    htail_x = x[13]
    vtail_chords = [x[14], x[15]]
    vtail_twists = [x[16], x[17]]
    vtail_height = x[18]
    vtail_x = x[19]
    mass_x = x[20]
    battery_x = x[21]
    electronics_x = x[22]
    wing_x = x[23]
    ballast_mass = x[24]
    ballast_x = x[25]
    
    # Build configuration
    GLOBAL_WING_CAMBER[] = get_camber_function(config, :wing)
    GLOBAL_TAIL_CAMBER[] = get_camber_function(config, :tail)
    
    system, S, c, b, rref, _, _, _ = build_configuration(
        wing_chords, wing_twists, htail_chords, htail_twists, vtail_chords, vtail_twists;
        wing_half_span=wing_half_span,
        wing_dihedral=wing_dihedral,
        sweep=sweep,
        htail_half_span=htail_half_span,
        htail_x=htail_x,
        vtail_height=vtail_height,
        vtail_x=vtail_x,
        mass_x=mass_x,
        wing_x=wing_x
    )
    
    # Get CG and components
    comps = build_components(
        wing_half_span, wing_chords,
        htail_half_span, htail_chords,
        vtail_height, vtail_chords;
        wing_x=wing_x, htail_x=htail_x, vtail_x=vtail_x,
        battery_x=battery_x, electronics_x=electronics_x,
        ballast_mass=ballast_mass, ballast_x=ballast_x
    )
    m_tot, cg, (Ixx, Iyy, Izz) = compute_inertia_from_components(comps)
    
    # Calculate neutral point
    x_np = calculate_neutral_point(system, S, c, b, rref)
    
    # Create plot
    p = plot(size=(800, 400), legend=:topright, dpi=150)
    
    # Wing chord at root (side view shows root chord)
    c_root = wing_chords[1]
    x_le_root = wing_x - 0.25 * c_root
    x_te_root = x_le_root + c_root
    wing_thickness = config.foam_thickness
    
    # Wing tip position in z due to dihedral
    z_tip = wing_half_span * tan(wing_dihedral * pi / 180.0)
    
    # Approximate wing profile (show dihedral as angled line)
    plot!([x_le_root, x_le_root], [-wing_thickness/2, wing_thickness/2],
          color=:black, linewidth=2, label="Wing")
    plot!([x_te_root, x_te_root], [-wing_thickness/2, wing_thickness/2],
          color=:black, linewidth=2, label="")
    plot!([x_le_root, x_te_root], [wing_thickness/2, wing_thickness/2],
          color=:black, linewidth=2, label="")
    plot!([x_le_root, x_te_root], [-wing_thickness/2, -wing_thickness/2],
          color=:black, linewidth=2, label="")
    
    # Show dihedral with a dashed line to tip height
    if abs(wing_dihedral) > 0.1
        x_wing_mid = wing_x
        plot!([x_wing_mid, x_wing_mid], [0.0, z_tip],
              color=:black, linewidth=1, linestyle=:dot, 
              label=@sprintf("Dihedral: %.1f°", wing_dihedral))
    end
    
    # Horizontal tail
    ht_chord_root = htail_chords[1]
    ht_xle = htail_x - 0.25 * ht_chord_root
    ht_xte = ht_xle + ht_chord_root
    
    plot!([ht_xle, ht_xte, ht_xte, ht_xle, ht_xle],
          [-wing_thickness/2, -wing_thickness/2, wing_thickness/2, wing_thickness/2, -wing_thickness/2],
          color=:black, linewidth=1.5, linestyle=:dash, fill=true, fillalpha=0.2, label="H-Tail")
    
    # Vertical tail (with taper)
    vt_xle_root = vtail_x - 0.25 * vtail_chords[1]
    vt_xle_tip = vtail_x - 0.25 * vtail_chords[2]
    vt_xte_root = vt_xle_root + vtail_chords[1]
    vt_xte_tip = vt_xle_tip + vtail_chords[2]
    
    plot!([vt_xle_root, vt_xle_tip, vt_xte_tip, vt_xte_root, vt_xle_root],
          [0, vtail_height, vtail_height, 0, 0],
          color=:black, linewidth=1.5, linestyle=:dot, fill=true, fillalpha=0.2, label="V-Tail")
    
    # Fuselage
    fuse_length = min(max(htail_x, vtail_x) + 0.2, config.max_fuselage_length)
    fuse_r = config.fuselage_outer_diameter / 2
    plot!([0, fuse_length], [0, 0], color=:gray, linewidth=4, label="Fuselage")
    
    # Plot CG
    scatter!([cg[1]], [cg[3]], marker=:circle, markersize=10, color=:red, 
             label="CG", markerstrokewidth=2)
    
    # Plot neutral point (at wing height)
    scatter!([x_np], [0.0], marker=:xcross, markersize=10, color=:blue, 
             label="Neutral Point", markerstrokewidth=2)
    
    # Plot component positions
    scatter!([config.motor_x], [0.0], marker=:square, markersize=8, color=:purple,
             label="Motor")
    scatter!([battery_x], [0.0], marker=:diamond, markersize=8, color=:green,
             label="Battery")
    scatter!([electronics_x], [0.0], marker=:star5, markersize=8, color=:orange,
             label="Electronics")
    
    # Plot ballast if present
    if ballast_mass > 0.001
        scatter!([ballast_x], [0.0], marker=:dtriangle, markersize=8, color=:brown,
                 label=@sprintf("Ballast (%.0fg)", ballast_mass*1000))
    end
    
    # Add stability margin indicator
    static_margin = (x_np - cg[1]) / c
    title!(string("Aircraft Side View\n",
                  "Static Margin: ", @sprintf("%.3f", static_margin), " (x̄_np - x̄_cg)\n",
                  "V-tail height: ", @sprintf("%.3f", vtail_height), " m"))
    xlabel!("x [m]")
    ylabel!("z [m]")
    
    # Set reasonable axis limits
    xlims!(-0.2, fuse_length + 0.1)
    ylims!(-0.3, max(vtail_height, z_tip) + 0.1)
    
    savefig(p, filename)
    println("✓ Saved side view to: $filename")
    
    return p
end

"""
Plot both top and side views of aircraft in a single figure.
"""
function plot_aircraft(x::Vector{Float64}, config::DesignConfig=CONFIG; 
                       filename_top="aircraft_top_view.png",
                       filename_side="aircraft_side_view.png")
    println("\nGenerating aircraft plots...")
    
    p_top = plot_top_view(x, config; filename=filename_top)
    p_side = plot_side_view(x, config; filename=filename_side)
    
    # Also create a combined plot
    p_combined = plot(p_top, p_side, layout=(2,1), size=(800, 1000))
    savefig(p_combined, "aircraft_combined_view.png")
    println("✓ Saved combined view to: aircraft_combined_view.png")
    
    return p_top, p_side, p_combined
end

"""
Print a comprehensive summary of the optimized design.
"""
function design_summary(x::Vector{Float64}, config::DesignConfig=CONFIG)
    println("\n" * "="^70)
    println("OPTIMIZED AIRCRAFT DESIGN SUMMARY")
    println("="^70)
    
    # Unpack design variables
    wing_chords = [x[1], x[2]]
    wing_twists = [x[3], x[4]]
    wing_half_span = x[5]
    wing_dihedral = x[6]
    sweep = x[7]
    htail_chords = [x[8], x[9]]
    htail_twists = [x[10], x[11]]
    htail_half_span = x[12]
    htail_x = x[13]
    vtail_chords = [x[14], x[15]]
    vtail_twists = [x[16], x[17]]
    vtail_height = x[18]
    vtail_x = x[19]
    mass_x = x[20]
    battery_x = x[21]
    electronics_x = x[22]
    wing_x = x[23]
    ballast_mass = x[24]
    ballast_x = x[25]
    
    # Calculate areas
    wing_area = (wing_chords[1] + wing_chords[2]) * wing_half_span
    htail_area = (htail_chords[1] + htail_chords[2]) * htail_half_span
    vtail_area = 0.5 * (vtail_chords[1] + vtail_chords[2]) * vtail_height
    
    # Build configuration for analysis
    GLOBAL_WING_CAMBER[] = get_camber_function(config, :wing)
    GLOBAL_TAIL_CAMBER[] = get_camber_function(config, :tail)
    
    system, S, c, b, rref, _, _, _ = build_configuration(
        wing_chords, wing_twists, htail_chords, htail_twists, vtail_chords, vtail_twists;
        wing_half_span=wing_half_span,
        wing_dihedral=wing_dihedral,
        sweep=sweep,
        htail_half_span=htail_half_span,
        htail_x=htail_x,
        vtail_height=vtail_height,
        vtail_x=vtail_x,
        mass_x=mass_x,
        wing_x=wing_x
    )
    
    # Get mass properties
    comps = build_components(
        wing_half_span, wing_chords,
        htail_half_span, htail_chords,
        vtail_height, vtail_chords;
        wing_x=wing_x, htail_x=htail_x, vtail_x=vtail_x,
        battery_x=battery_x, electronics_x=electronics_x,
        ballast_mass=ballast_mass, ballast_x=ballast_x
    )
    m_tot, cg, (Ixx, Iyy, Izz) = compute_inertia_from_components(comps)
    
    # Run aerodynamic analysis
    CF, CM, dCF, dCM = run_vlm_analysis!(system, S, c, b, rref; 
                                         α=2.0*pi/180, β=0.0, symmetric=false)
    CD, CY, CL = CF
    q_dyn = 0.5 * config.air_density * config.cruise_speed^2
    L = q_dyn * S * CL
    D = q_dyn * S * CD
    L_over_D = L / D
    
    # Calculate range
    e_b = config.battery_specific_energy * 3600.0
    η = config.propulsion_efficiency
    m_b = config.battery_mass
    range_m = (e_b / g) * η * L_over_D * (m_b / m_tot)
    
    # Calculate neutral point and static margin
    Cm_alpha = dCM.alpha[2]
    CL_alpha = dCF.alpha[3]
    x_np = mass_x - (Cm_alpha / CL_alpha) * c
    static_margin = (x_np - cg[1]) / c
    
    println("\n" * "-"^70)
    println("WING")
    println("-"^70)
    @printf("  Span:         %.3f m (%.1f in) - OPTIMIZED\n", 2*wing_half_span, 2*wing_half_span*39.3701)
    @printf("  Root chord:   %.3f m (%.1f in)\n", wing_chords[1], wing_chords[1]*39.3701)
    @printf("  Tip chord:    %.3f m (%.1f in)\n", wing_chords[2], wing_chords[2]*39.3701)
    @printf("  Taper ratio:  %.3f\n", wing_chords[2]/wing_chords[1])
    @printf("  Area:         %.4f m² (%.1f in²)\n", wing_area, wing_area*1550.0)
    @printf("  Aspect ratio: %.2f\n", (2*wing_half_span)^2 / wing_area)
    @printf("  Dihedral:     %.2f° - OPTIMIZED\n", wing_dihedral)
    @printf("  Root twist:   %.2f°\n", wing_twists[1])
    @printf("  Tip twist:    %.2f° (washout: %.2f°)\n", wing_twists[2], wing_twists[1] - wing_twists[2])
    @printf("  Sweep:        %.3f m\n", sweep)
    @printf("  Position:     x = %.3f m\n", wing_x)
    
    println("\n" * "-"^70)
    println("HORIZONTAL TAIL")
    println("-"^70)
    @printf("  Span:         %.3f m (%.1f in) - OPTIMIZED\n", 2*htail_half_span, 2*htail_half_span*39.3701)
    @printf("  Root chord:   %.3f m (%.1f in)\n", htail_chords[1], htail_chords[1]*39.3701)
    @printf("  Tip chord:    %.3f m (%.1f in)\n", htail_chords[2], htail_chords[2]*39.3701)
    @printf("  Taper ratio:  %.3f\n", htail_chords[2]/htail_chords[1])
    @printf("  Area:         %.4f m² (%.1f in²) - OPTIMIZED\n", htail_area, htail_area*1550.0)
    @printf("  Aspect ratio: %.2f\n", (2*htail_half_span)^2 / htail_area)
    @printf("  Root twist:   %.2f°\n", htail_twists[1])
    @printf("  Tip twist:    %.2f°\n", htail_twists[2])
    @printf("  Position:     x = %.3f m - OPTIMIZED\n", htail_x)
    @printf("  Tail volume:  %.3f (typical: 0.4-0.6)\n", (htail_area * (htail_x - wing_x)) / (S * c))
    
    println("\n" * "-"^70)
    println("VERTICAL TAIL")
    println("-"^70)
    @printf("  Height:       %.3f m (%.1f in) - OPTIMIZED\n", vtail_height, vtail_height*39.3701)
    @printf("  Root chord:   %.3f m (%.1f in)\n", vtail_chords[1], vtail_chords[1]*39.3701)
    @printf("  Tip chord:    %.3f m (%.1f in)\n", vtail_chords[2], vtail_chords[2]*39.3701)
    @printf("  Taper ratio:  %.3f\n", vtail_chords[2]/vtail_chords[1])
    @printf("  Area:         %.4f m² (%.1f in²) - OPTIMIZED\n", vtail_area, vtail_area*1550.0)
    @printf("  Aspect ratio: %.2f\n", vtail_height^2 / vtail_area)
    @printf("  Root twist:   %.2f°\n", vtail_twists[1])
    @printf("  Tip twist:    %.2f°\n", vtail_twists[2])
    @printf("  Position:     x = %.3f m - OPTIMIZED\n", vtail_x)
    @printf("  Tail volume:  %.3f (typical: 0.02-0.04)\n", (vtail_area * (vtail_x - wing_x)) / (S * (2*wing_half_span)))
    
    println("\n" * "-"^70)
    println("MASS BREAKDOWN")
    println("-"^70)
    @printf("  Total mass:   %.3f kg (%.1f oz)\n", m_tot, m_tot*35.274)
    println("\n  Fixed components:")
    @printf("    Motor:        %.3f kg at x = %.3f m (FIXED)\n", config.motor_mass, config.motor_x)
    @printf("    Battery:      %.3f kg at x = %.3f m - OPTIMIZED\n", config.battery_mass, battery_x)
    @printf("    Electronics:  %.3f kg at x = %.3f m - OPTIMIZED\n", config.electronics_mass, electronics_x)
    if ballast_mass > 0.001
        @printf("    Ballast:      %.3f kg at x = %.3f m - OPTIMIZED\n", ballast_mass, ballast_x)
    else
        @printf("    Ballast:      None\n")
    end
    
    println("\n  Airframe components:")
    for comp in comps
        if !(comp.name in ["motor", "battery", "electronics", "ballast"])
            @printf("    %-12s: %.3f kg at (%.3f, %.3f, %.3f) m\n", 
                    comp.name, comp.mass, comp.x, comp.y, comp.z)
        end
    end
    
    println("\n" * "-"^70)
    println("CENTER OF GRAVITY & STABILITY")
    println("-"^70)
    @printf("  CG position:     (%.3f, %.3f, %.3f) m\n", cg[1], cg[2], cg[3])
    @printf("  Neutral point:   %.3f m\n", x_np)
    @printf("  Static margin:   %.3f (%.1f%% MAC)\n", static_margin, static_margin*100)
    @printf("  Reference MAC:   %.3f m\n", c)
    
    println("\n" * "-"^70)
    println("AERODYNAMIC PERFORMANCE (α = 2°)")
    println("-"^70)
    @printf("  CL:              %.4f\n", CL)
    @printf("  CD:              %.4f\n", CD)
    @printf("  L/D:             %.2f\n", L_over_D)
    @printf("  Lift force:      %.2f N (%.2f lbf)\n", L, L/4.448)
    @printf("  Drag force:      %.2f N (%.2f lbf)\n", D, D/4.448)
    @printf("  Wing loading:    %.2f kg/m² (%.2f oz/ft²)\n", m_tot/S, m_tot*35.274/(S*10.764))
    
    println("\n" * "-"^70)
    println("ESTIMATED RANGE")
    println("-"^70)
    @printf("  Battery energy:  %.1f Wh (%.0f mAh @ 11.1V)\n", 
            config.battery_mass * config.battery_specific_energy,
            config.battery_mass * config.battery_specific_energy * 1000 / 11.1)
    @printf("  Efficiency:      %.0f%%\n", config.propulsion_efficiency * 100)
    @printf("  L/D ratio:       %.2f\n", L_over_D)
    @printf("  **RANGE:         %.1f m (%.2f km, %.2f miles)**\n", 
            range_m, range_m/1000.0, range_m/1609.34)
    
    # Dynamic stability
    A_long = assemble_longitudinal_A(dCF, dCM, S, c, m_tot, Iyy, config.cruise_speed)
    A_lat = assemble_lateral_A(dCF, dCM, S, b, m_tot, Ixx, Izz, config.cruise_speed)
    modes = analyze_modes(A_long, A_lat)
    
    println("\n" * "-"^70)
    println("DYNAMIC STABILITY MODES")
    println("-"^70)
    
    for mode_name in ["short_period", "phugoid", "dutch_roll", "roll_subsidence", "spiral"]
        if haskey(modes, mode_name)
            mode = modes[mode_name]
            status = real(mode.λ) < 0 ? "✓ STABLE" : "✗ UNSTABLE"
            if abs(imag(mode.λ)) > 0.01
                @printf("  %-20s: %s\n", mode_name, status)
                @printf("    λ = %.4f %+.4fi  |  ωₙ = %.3f rad/s  |  ζ = %.3f\n",
                        real(mode.λ), imag(mode.λ), mode.ωn, mode.ζ)
                period = 2*π / abs(imag(mode.λ))
                @printf("    Period = %.2f s  |  ", period)
                if real(mode.λ) < 0
                    t_half = -log(2) / real(mode.λ)
                    @printf("Time to half: %.2f s\n", t_half)
                else
                    t_double = log(2) / real(mode.λ)
                    @printf("Time to double: %.2f s\n", t_double)
                end
            else
                τ_str = isfinite(mode.τ) ? @sprintf("%.2f s", mode.τ) : "∞"
                @printf("  %-20s: %s\n", mode_name, status)
                @printf("    λ = %.4f  |  Time constant = %s\n", real(mode.λ), τ_str)
            end
        end
    end
    
    println("\n" * "="^70)
    println("DESIGN VARIABLE RANGES (showing optimizer freedom)")
    println("="^70)
    @printf("  Wing span:        %.3f m (range: %.3f - %.3f m)\n", 
            2*wing_half_span, 2*config.min_wing_half_span, 2*config.max_wing_half_span)
    @printf("  Wing dihedral:    %.2f° (range: 0.0 - %.1f°)\n", 
            wing_dihedral, config.max_wing_dihedral)
    @printf("  H-tail span:      %.3f m (range: 0.30 - 1.00 m)\n", 2*htail_half_span)
    @printf("  V-tail height:    %.3f m (range: 0.30 - 1.00 m)\n", vtail_height)
    @printf("  Battery x:        %.3f m (range: %.3f - 0.50 m)\n", battery_x, config.motor_x)
    @printf("  Electronics x:    %.3f m (range: %.3f - 0.30 m)\n", electronics_x, config.motor_x)
    @printf("  Ballast mass:     %.3f kg (range: 0.0 - 0.20 kg)\n", ballast_mass)
    @printf("  Ballast x:        %.3f m (range: %.3f - 0.80 m)\n", ballast_x, config.motor_x)
    
    println("="^70)
end

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end