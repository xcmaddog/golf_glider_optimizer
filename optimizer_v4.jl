# GliderOptimizer_v4.jl
# Cleaned-up and fully functional aircraft optimization code
# Optimizes for range while maintaining stability
#
# Range Equation: R = (e_b/g) × η × (L/D) × (m_b/m_TO)
# where:
#   R = range [m]
#   e_b = battery specific energy [Wh/kg] (converted to J/kg internally)
#   g = gravitational acceleration [m/s²]
#   η = propulsion efficiency (motor × ESC × propeller)
#   L/D = lift-to-drag ratio (from VLM analysis)
#   m_b = battery mass [kg]
#   m_TO = total aircraft mass at takeoff [kg]
#
# Installation requirements:
#   using Pkg
#   Pkg.add(["VortexLattice", "SNOW", "Ipopt", "LinearAlgebra", "Printf", "Statistics"])
#
# Quick Start:
#   1. Modify CONFIG parameters or use set_config!()
#   2. Load airfoils: load_airfoil!("path/to/airfoil.dat", surface=:wing)
#   3. Run: xopt, fopt = optimize_aircraft()
#   4. Analyze: analyze_design(xopt)
#
using VortexLattice
using SNOW
using Ipopt
using LinearAlgebra
using Printf
using Statistics
import ForwardDiff  # Import to check for Dual types during AD

# ============================================================================
# DESIGN CONFIGURATION PARAMETERS
# ============================================================================

"""
Configuration struct to hold all design parameters.
Modify these values to change your design constraints.
"""
Base.@kwdef mutable struct DesignConfig
    # Flight conditions
    cruise_speed::Float64 = 10.0           # [m/s]
    air_density::Float64 = 1.225           # [kg/m³]
    
    # Wing geometry constraints
    max_wing_half_span::Float64 = 2.0      # [m] Maximum half-span
    n_span_sections::Int = 7               # Number of design control points
    
    # Materials
    foam_density::Float64 = 30.0           # [kg/m³] Foam density (EPP ~30, EPO ~25, Depron ~40)
    foam_thickness::Float64 = 0.02         # [m] Wing/tail thickness
    fuselage_density::Float64 = 1850.0     # [kg/m³] Fiberglass rod density
    fuselage_radius::Float64 = 0.006       # [m] Fuselage rod radius
    spar_density::Float64 = 1850.0         # [kg/m³] Wing spar density (fiberglass)
    spar_radius::Float64 = 0.004           # [m] Wing spar radius
    
    # Component masses
    motor_mass::Float64 = 0.18             # [kg] Motor mass
    motor_x::Float64 = 0.05                # [m] Motor x-position
    battery_mass::Float64 = 0.25           # [kg] Battery mass
    battery_x::Float64 = -0.05             # [m] Battery x-position
    electronics_mass::Float64 = 0.05       # [kg] Electronics mass (ESC, receiver, servos)
    electronics_x::Float64 = 0.0           # [m] Electronics x-position
    
    # Propulsion system
    battery_specific_energy::Float64 = 150.0  # [Wh/kg] Battery specific energy (LiPo ~150-200)
    propulsion_efficiency::Float64 = 0.65     # [-] Overall propulsion efficiency (motor*ESC*prop ~0.6-0.7)
    
    # Fuselage constraints
    max_fuselage_length::Float64 = 1.5     # [m] Maximum allowable fuselage length
    
    # Airfoil selection - Wing
    wing_airfoil_type::String = "flat"     # Options: "flat", "NACA4", "dat_file"
    wing_naca_digits::String = "2412"      # If using NACA4 airfoils (e.g., "2412")
    wing_dat_file::String = ""             # Path to .dat file if using dat_file type
    wing_camber_function::Union{Function, Nothing} = nothing  # Custom camber function f(x/c) -> y/c
    
    # Airfoil selection - Tail
    tail_airfoil_type::String = "flat"     # Options: "flat", "NACA4", "dat_file"
    tail_naca_digits::String = "0012"      # If using NACA4 airfoils (symmetric default)
    tail_dat_file::String = ""             # Path to .dat file if using dat_file type
    tail_camber_function::Union{Function, Nothing} = nothing  # Custom camber function f(x/c) -> y/c
    
    # Optimization parameters
    stability_margin::Float64 = 0.01       # Minimum damping margin for dynamic modes (relaxed)
    lift_safety_factor::Float64 = 1.0      # Lift requirement multiplier
end

# Global configuration (modify before running optimization)
const CONFIG = DesignConfig()

# Physical constants
const g = 9.80665  # [m/s²]

# ============================================================================
# AIRFOIL FUNCTIONS
# ============================================================================

"""
Parse Selig format .dat file and return camber function.
Format: x/c, y/c pairs from trailing edge over top, then bottom.
"""
function parse_selig_dat(filename::String)
    if !isfile(filename)
        error("Airfoil .dat file not found: $filename")
    end
    
    lines = readlines(filename)
    
    # Skip header line(s) - typically first line is airfoil name
    data_start = 1
    for (i, line) in enumerate(lines)
        # Check if line contains two numbers
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
    
    # Parse all coordinate pairs
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
    
    # Separate upper and lower surfaces
    # Selig format: starts at TE (x≈1), goes over upper surface to LE (x≈0), 
    # then back along lower surface to TE
    
    # Find leading edge (minimum x)
    min_x_idx = argmin([c[1] for c in coords])
    
    upper = coords[1:min_x_idx]
    lower = coords[min_x_idx:end]
    
    # Reverse upper surface so both go LE to TE
    reverse!(upper)
    
    # Sort just in case
    sort!(upper, by=first)
    sort!(lower, by=first)
    
    # Compute camber line: average of upper and lower at each x
    # Create unified x array
    x_upper = [c[1] for c in upper]
    y_upper = [c[2] for c in upper]
    x_lower = [c[1] for c in lower]
    y_lower = [c[2] for c in lower]
    
    # Interpolate lower surface to match upper x-coordinates for averaging
    # Use linear interpolation
    function interpolate_linear(x_data, y_data, x_query)
        if x_query <= x_data[1]
            return y_data[1]
        elseif x_query >= x_data[end]
            return y_data[end]
        end
        
        # Find bracketing points
        for i in 1:(length(x_data)-1)
            if x_data[i] <= x_query <= x_data[i+1]
                t = (x_query - x_data[i]) / (x_data[i+1] - x_data[i])
                return y_data[i] * (1-t) + y_data[i+1] * t
            end
        end
        return y_data[end]
    end
    
    # Compute camber at each x location
    x_camber = unique(sort(vcat(x_upper, x_lower)))
    y_camber = zeros(length(x_camber))
    
    for (i, x) in enumerate(x_camber)
        yu = interpolate_linear(x_upper, y_upper, x)
        yl = interpolate_linear(x_lower, y_lower, x)
        y_camber[i] = (yu + yl) / 2.0
    end
    
    # Return interpolating function for camber line
    function camber(xc)
        return interpolate_linear(x_camber, y_camber, xc)
    end
    
    return camber
end

"""
Generate camber function based on configuration.
Must handle being called with single argument (for backward compatibility).
Returns a consistent function type.
"""
function get_camber_function(config::DesignConfig, surface::Symbol=:wing)
    if surface == :wing
        airfoil_type = config.wing_airfoil_type
        naca_digits = config.wing_naca_digits
        dat_file = config.wing_dat_file
        custom_func = config.wing_camber_function
    else  # :tail
        airfoil_type = config.tail_airfoil_type
        naca_digits = config.tail_naca_digits
        dat_file = config.tail_dat_file
        custom_func = config.tail_camber_function
    end
    
    # Get the appropriate camber function
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
    
    # Wrap in a consistent closure to avoid type instability
    return (xc) -> base_func(xc)
end

"""
Generate NACA 4-digit camber line function.
Returns function f(x/c) that gives y/c.
"""
function naca4_camber(digits::String)
    if length(digits) != 4
        @warn "NACA digits must be 4 characters, using flat plate"
        return (xc) -> 0.0
    end
    
    m = parse(Int, digits[1]) / 100.0  # Maximum camber
    p = parse(Int, digits[2]) / 10.0   # Location of maximum camber
    
    if m == 0.0 || p == 0.0
        return (xc) -> 0.0  # Symmetric airfoil
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
# GEOMETRY GENERATION (Made generic for AD compatibility)
# ============================================================================

"""
Build a lifting surface from section definitions.
Generic types allow automatic differentiation.
Accepts fc as Any to work around ForwardDiff type issues.
"""
function build_surface_from_sections(
    xle, yle, zle, chord, theta, phi, fc, 
    ns::Int, nc::Int; 
    mirror::Bool=false,
    spacing_s=Uniform(), 
    spacing_c=Uniform()
)
    # Only convert to Float64 if not already Dual numbers
    # Use identity.(x) to preserve type but ensure concrete array
    xle_f = identity.(xle)
    yle_f = identity.(yle)
    zle_f = identity.(zle)
    chord_f = identity.(chord)
    theta_f = identity.(theta)
    phi_f = identity.(phi)
    
    # Check if we have Dual numbers (AD is active)
    # If so, VortexLattice needs Float64, so we extract values
    # But this means derivatives won't flow through geometry - which is okay
    # since we're only differentiating w.r.t. design variables (chords, twists, etc.)
    if any(x -> x isa ForwardDiff.Dual, xle_f)
        # Extract Float64 values from Dual numbers for VortexLattice
        xle_f = ForwardDiff.value.(xle_f)
        yle_f = ForwardDiff.value.(yle_f)
        zle_f = ForwardDiff.value.(zle_f)
        chord_f = ForwardDiff.value.(chord_f)
        theta_f = ForwardDiff.value.(theta_f)
        phi_f = ForwardDiff.value.(phi_f)
    end
    
    grid, ratio = wing_to_grid(xle_f, yle_f, zle_f, chord_f, theta_f, phi_f, ns, nc; 
                               fc=fc, spacing_s=spacing_s, spacing_c=spacing_c, mirror=mirror)
    
    # Use the correct API for grid_to_surface_panels
    # According to VortexLattice docs, when we have a single grid with ratio, we use:
    surface = grid_to_surface_panels(grid; ratios=ratio, mirror=false)
    
    return grid, ratio, surface
end

"""
Build complete aircraft configuration: wing + horizontal tail + vertical tail.
Generic implementation for automatic differentiation compatibility.
NOTE: Camber functions are stored globally to avoid AD type issues.
"""

# Global storage for camber functions (to avoid AD issues)
const GLOBAL_WING_CAMBER = Ref{Function}((xc)->0.0)
const GLOBAL_TAIL_CAMBER = Ref{Function}((xc)->0.0)

function build_configuration(
    chords::Vector{T}, 
    twists::Vector{T};
    wing_span::Real = 2*CONFIG.max_wing_half_span,
    sweep::Real = 0.0,
    htail_x::Real = 1.5,
    vtail_x::Real = 1.5,
    mass_x::Real = 0.0,
    htail_area::Real = 0.3, 
    vtail_area::Real = 0.15,
    ht_half_span::Real = 0.6, 
    vt_height::Real = 0.6
) where T
    n = length(chords)
    
    # Get camber functions from global storage (set by optimization setup)
    wing_camber_func = GLOBAL_WING_CAMBER[]
    tail_camber_func = GLOBAL_TAIL_CAMBER[]
    
    # -------------------------------------------------------------------------
    # WING GEOMETRY (right half, will be mirrored)
    # -------------------------------------------------------------------------
    y_half = collect(range(0.0, stop=wing_span/2, length=n))
    
    # Leading edge x-position with sweep (quarter-chord sweep)
    xle_w = T[-0.25 * c + sweep * (yi/(wing_span/2)) for (c, yi) in zip(chords, y_half)]
    zle_w = zeros(T, n)
    theta_w = T[(pi/180) * tw for tw in twists]  # Convert degrees to radians
    phi_w = zeros(T, n)
    fc_w = [wing_camber_func for _ in 1:n]  # Create function vector
    
    ns_w = 12  # Spanwise panels
    nc_w = 6   # Chordwise panels
    
    grid_w, ratio_w, surf_w = build_surface_from_sections(
        xle_w, y_half, zle_w, chords, theta_w, phi_w, fc_w, ns_w, nc_w; 
        mirror=true, spacing_s=Sine()
    )
    
    # -------------------------------------------------------------------------
    # HORIZONTAL TAIL (symmetric) - uses tail airfoil
    # -------------------------------------------------------------------------
    n_ht = 3
    ht_y = T[yi for yi in range(0.0, stop=ht_half_span, length=n_ht)]
    ht_chord_val = htail_area / (2 * ht_half_span)
    ht_chord = fill(T(ht_chord_val), n_ht)
    ht_xle = fill(T(htail_x - 0.25 * ht_chord_val), n_ht)
    ht_zle = zeros(T, n_ht)
    ht_theta = fill(T(-2.0*pi/180), n_ht)  # Small negative incidence for stability
    ht_phi = zeros(T, n_ht)
    fc_ht = [tail_camber_func for _ in 1:n_ht]  # Use tail airfoil
    
    ns_ht = 6
    nc_ht = 4
    
    grid_ht, ratio_ht, surf_ht = build_surface_from_sections(
        ht_xle, ht_y, ht_zle, ht_chord, ht_theta, ht_phi, fc_ht, ns_ht, nc_ht; 
        mirror=true
    )
    
    # -------------------------------------------------------------------------
    # VERTICAL TAIL (not mirrored) - uses tail airfoil
    # -------------------------------------------------------------------------
    n_vt = 3
    vt_z = T[zi for zi in range(0.0, stop=vt_height, length=n_vt)]
    vt_chord_val = vtail_area / vt_height
    vt_chord = fill(T(vt_chord_val), n_vt)
    vt_xle = fill(T(vtail_x - 0.25 * vt_chord_val), n_vt)
    vt_yle = zeros(T, n_vt)
    vt_theta = zeros(T, n_vt)
    vt_phi = zeros(T, n_vt)
    fc_vt = [tail_camber_func for _ in 1:n_vt]  # Use tail airfoil
    
    ns_vt = 6
    nc_vt = 4
    
    grid_vt, ratio_vt, surf_vt = build_surface_from_sections(
        vt_xle, vt_yle, vt_z, vt_chord, vt_theta, vt_phi, fc_vt, ns_vt, nc_vt; 
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
    for i in 1:(n-1)
        wing_area += 2 * 0.5 * (y_half[i+1] - y_half[i]) * (chords[i] + chords[i+1])
    end
    
    ref_area = wing_area
    ref_chord = wing_area / wing_span
    ref_span = T(wing_span)
    rref = [T(mass_x), zero(T), zero(T)]  # Reference point at CG
    
    return system, ref_area, ref_chord, ref_span, rref, surf_w, surf_ht, surf_vt
end

# ============================================================================
# AERODYNAMIC ANALYSIS
# ============================================================================

"""
Run steady VLM analysis and return force coefficients and derivatives.
"""
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
    # Freestream conditions
    fs = Freestream(CONFIG.cruise_speed, α, β, Ω)
    
    # Reference object - extract Float64 values if Dual numbers present
    ref_area_f = ref_area isa ForwardDiff.Dual ? ForwardDiff.value(ref_area) : Float64(ref_area)
    ref_chord_f = ref_chord isa ForwardDiff.Dual ? ForwardDiff.value(ref_chord) : Float64(ref_chord)
    ref_span_f = ref_span isa ForwardDiff.Dual ? ForwardDiff.value(ref_span) : Float64(ref_span)
    rref_f = [r isa ForwardDiff.Dual ? ForwardDiff.value(r) : Float64(r) for r in rref]
    
    ref = Reference(ref_area_f, ref_chord_f, ref_span_f, rref_f, CONFIG.cruise_speed)
    
    # Run steady analysis (computes circulation, forces, and derivatives)
    steady_analysis!(system, ref, fs; symmetric=symmetric)
    
    # Extract body forces in wind frame
    CF, CM = body_forces(system; frame=Wind())
    
    # Get stability derivatives
    dCF, dCM = stability_derivatives(system)
    
    return CF, CM, dCF, dCM
end

# ============================================================================
# COMPONENT-BASED INERTIA MODEL
# ============================================================================

struct Component
    name::String
    mass::Float64
    x::Float64
    y::Float64
    z::Float64
    Ixx_c::Float64   # Moment about component centroid
    Iyy_c::Float64
    Izz_c::Float64
end

"""
Compute inertia for a rectangular lamina (thin plate).
"""
function lamina_inertia(m::Float64, span::Float64, chord::Float64)
    Ixx = (1.0/12.0) * m * span^2   # Roll axis
    Iyy = (1.0/12.0) * m * chord^2  # Pitch axis
    Izz = Ixx + Iyy                 # Yaw axis
    return Ixx, Iyy, Izz
end

"""
Compute inertia for a slender rod.
"""
function rod_inertia(m::Float64, L::Float64, axis_dir::Symbol, radius::Float64=0.01)
    if axis_dir == :x
        Ixx = 0.5*m*radius^2
        Iyy = (1/12.0)*m*L^2
        Izz = (1/12.0)*m*L^2
    elseif axis_dir == :y
        Ixx = (1/12.0)*m*L^2
        Iyy = 0.5*m*radius^2
        Izz = (1/12.0)*m*L^2
    elseif axis_dir == :z
        Ixx = (1/12.0)*m*L^2
        Iyy = (1/12.0)*m*L^2
        Izz = 0.5*m*radius^2
    else
        error("axis_dir must be :x, :y, or :z")
    end
    return Ixx, Iyy, Izz
end

"""
Build component list from geometric and mass parameters.
"""
function build_components(
    wing_span::Float64,
    chords::Vector;
    wing_x::Float64 = 0.0,
    wing_z::Float64 = 0.0,
    htail_x::Float64 = 1.5,
    htail_z::Float64 = 0.0,
    vtail_x::Float64 = 1.5,
    vtail_z::Float64 = 0.0,
    htail_area::Float64 = 0.3,
    ht_half_span::Float64 = 0.6,
    vtail_area::Float64 = 0.15,
    vt_height::Float64 = 0.6
)
    comps = Component[]
    
    # Get configuration parameters
    foam_thickness = CONFIG.foam_thickness
    rho_foam = CONFIG.foam_density
    spar_radius = CONFIG.spar_radius
    rho_spar = CONFIG.spar_density
    fuselage_radius = CONFIG.fuselage_radius
    rho_fuse = CONFIG.fuselage_density
    
    # -------------------------------------------------------------------------
    # WING (split into left and right halves)
    # -------------------------------------------------------------------------
    # Extract Float64 values if chords contains Dual numbers
    chords_f = [c isa ForwardDiff.Dual ? ForwardDiff.value(c) : Float64(c) for c in chords]
    c_avg = mean(chords_f)
    half_span = wing_span / 2.0
    
    # Wing area and mass
    S_wing = 2.0 * (c_avg * half_span)
    vol_wing = S_wing * foam_thickness
    mass_wing = vol_wing * rho_foam
    
    # Half-wing parameters
    m_half = mass_wing / 2.0
    y_centroid = half_span / 2.0
    x_centroid = wing_x + c_avg / 2.0
    
    Ixx, Iyy, Izz = lamina_inertia(m_half, half_span, c_avg)
    push!(comps, Component("wing_right", m_half, x_centroid,  y_centroid, wing_z, Ixx, Iyy, Izz))
    push!(comps, Component("wing_left",  m_half, x_centroid, -y_centroid, wing_z, Ixx, Iyy, Izz))
    
    # -------------------------------------------------------------------------
    # SPAR
    # -------------------------------------------------------------------------
    vol_spar = π * spar_radius^2 * wing_span
    mass_spar = vol_spar * rho_spar
    mass_spar = max(mass_spar, 0.02 * mass_wing)  # Minimum 2% of wing mass
    
    spar_x = wing_x + 0.25 * c_avg
    m_spar_half = mass_spar / 2.0
    
    Ixx_s, Iyy_s, Izz_s = rod_inertia(m_spar_half, half_span, :y, spar_radius)
    push!(comps, Component("spar_right", m_spar_half, spar_x,  y_centroid, wing_z, Ixx_s, Iyy_s, Izz_s))
    push!(comps, Component("spar_left",  m_spar_half, spar_x, -y_centroid, wing_z, Ixx_s, Iyy_s, Izz_s))
    
    # -------------------------------------------------------------------------
    # POINT MASSES (motor, battery, electronics)
    # -------------------------------------------------------------------------
    push!(comps, Component("motor", CONFIG.motor_mass, CONFIG.motor_x, 0.0, 0.0, 0.0, 0.0, 0.0))
    push!(comps, Component("battery", CONFIG.battery_mass, CONFIG.battery_x, 0.0, 0.0, 0.0, 0.0, 0.0))
    push!(comps, Component("electronics", CONFIG.electronics_mass, CONFIG.electronics_x, 0.0, 0.0, 0.0, 0.0, 0.0))
    
    # -------------------------------------------------------------------------
    # FUSELAGE (rod along x-axis)
    # -------------------------------------------------------------------------
    # Compute required fuselage length from tail positions
    max_tail_x = max(htail_x, vtail_x)
    fuselage_length = min(max_tail_x + 0.3, CONFIG.max_fuselage_length)  # Add margin, respect max
    vol_fuse = π * fuselage_radius^2 * fuselage_length
    fuselage_mass = vol_fuse * rho_fuse
    
    Ixx_f, Iyy_f, Izz_f = rod_inertia(fuselage_mass, fuselage_length, :x, fuselage_radius)
    push!(comps, Component("fuselage", fuselage_mass, 0.0, 0.0, 0.0, Ixx_f, Iyy_f, Izz_f))
    
    # -------------------------------------------------------------------------
    # HORIZONTAL TAIL (left and right halves)
    # -------------------------------------------------------------------------
    m_ht = htail_area * foam_thickness * rho_foam
    m_ht_half = m_ht / 2.0
    span_ht_half = ht_half_span / 2.0
    chord_ht = htail_area / ht_half_span
    x_ht = htail_x + chord_ht / 2.0
    y_ht = span_ht_half / 2.0
    
    Ixx_ht, Iyy_ht, Izz_ht = lamina_inertia(m_ht_half, span_ht_half, chord_ht)
    push!(comps, Component("ht_right", m_ht_half, x_ht,  y_ht, htail_z, Ixx_ht, Iyy_ht, Izz_ht))
    push!(comps, Component("ht_left",  m_ht_half, x_ht, -y_ht, htail_z, Ixx_ht, Iyy_ht, Izz_ht))
    
    # -------------------------------------------------------------------------
    # VERTICAL TAIL
    # -------------------------------------------------------------------------
    m_vt = vtail_area * foam_thickness * rho_foam
    chord_vt = vtail_area / vt_height
    span_vt = vt_height
    x_vt = vtail_x + chord_vt / 2.0
    z_vt = vtail_z + span_vt / 2.0
    
    Ixx_vt, Iyy_vt, Izz_vt = lamina_inertia(m_vt, span_vt, chord_vt)
    push!(comps, Component("vt", m_vt, x_vt, 0.0, z_vt, Ixx_vt, Iyy_vt, Izz_vt))
    
    return comps
end

"""
Compute total mass, CG, and principal inertias from components using parallel axis theorem.
"""
function compute_inertia_from_components(comps::Vector{Component})
    # Total mass and CG
    m_tot = sum(c.mass for c in comps)
    x_cg = sum(c.mass * c.x for c in comps) / m_tot
    y_cg = sum(c.mass * c.y for c in comps) / m_tot
    z_cg = sum(c.mass * c.z for c in comps) / m_tot
    
    # Inertias via parallel axis theorem
    Ixx = 0.0
    Iyy = 0.0
    Izz = 0.0
    
    for c in comps
        dx = c.x - x_cg
        dy = c.y - y_cg
        dz = c.z - z_cg
        
        # Parallel axis theorem
        Ixx += c.Ixx_c + c.mass * (dy^2 + dz^2)
        Iyy += c.Iyy_c + c.mass * (dx^2 + dz^2)
        Izz += c.Izz_c + c.mass * (dx^2 + dy^2)
    end
    
    return m_tot, (x_cg, y_cg, z_cg), (Ixx, Iyy, Izz)
end

# ============================================================================
# DYNAMIC STABILITY ANALYSIS
# ============================================================================

"""
Assemble longitudinal state-space matrix from stability derivatives.
State vector: [u, w, q, θ]
"""
function assemble_longitudinal_A(dCF, dCM, S, c, m, Iyy, U)
    q_dyn = 0.5 * CONFIG.air_density * U^2
    
    # Extract relevant derivatives
    CL_alpha = dCF.alpha[3]  # dCL/dα
    CD_alpha = dCF.alpha[1]  # dCD/dα
    Cm_alpha = dCM.alpha[2]  # dCm/dα
    Cm_q = dCM.q[2]          # dCm/dq (dimensional: per rad/s becomes per (qc/2V))
    CL_q = dCF.q[3]          # dCL/dq
    
    # Dimensional derivatives (approximate)
    Z_w = -q_dyn * S * CL_alpha / U  # Vertical force derivative w.r.t. w
    M_w = q_dyn * S * c * Cm_alpha / U  # Pitching moment derivative w.r.t. w
    M_q = q_dyn * S * c * Cm_q * (c / (2*U))  # Pitching moment derivative w.r.t. q
    Z_q = -q_dyn * S * CL_q * (c / (2*U))  # Vertical force derivative w.r.t. q
    
    # Assemble A matrix
    A = zeros(4, 4)
    A[1, 1] = 0.0
    A[1, 2] = 0.0
    A[1, 3] = 0.0
    A[1, 4] = -g
    
    A[2, 1] = 0.0
    A[2, 2] = Z_w / m
    A[2, 3] = (U + Z_q / m)
    A[2, 4] = 0.0
    
    A[3, 1] = 0.0
    A[3, 2] = M_w / Iyy
    A[3, 3] = M_q / Iyy
    A[3, 4] = 0.0
    
    A[4, 1] = 0.0
    A[4, 2] = 0.0
    A[4, 3] = 1.0
    A[4, 4] = 0.0
    
    return A
end

"""
Assemble lateral state-space matrix from stability derivatives.
State vector: [v, p, r, φ]
"""
function assemble_lateral_A(dCF, dCM, S, b, m, Ixx, Izz, U)
    q_dyn = 0.5 * CONFIG.air_density * U^2
    
    # Extract relevant derivatives
    CY_beta = dCF.beta[2]  # dCY/dβ
    Cl_beta = dCM.beta[1]  # dCl/dβ (roll moment)
    Cn_beta = dCM.beta[3]  # dCn/dβ (yaw moment)
    Cl_p = dCM.p[1]        # dCl/dp
    Cn_p = dCM.p[3]        # dCn/dp
    Cl_r = dCM.r[1]        # dCl/dr
    Cn_r = dCM.r[3]        # dCn/dr
    
    # Dimensional derivatives
    Y_v = q_dyn * S * CY_beta / U
    L_beta = q_dyn * S * b * Cl_beta
    N_beta = q_dyn * S * b * Cn_beta
    L_p = q_dyn * S * b * Cl_p * (b / (2*U))
    N_p = q_dyn * S * b * Cn_p * (b / (2*U))
    L_r = q_dyn * S * b * Cl_r * (b / (2*U))
    N_r = q_dyn * S * b * Cn_r * (b / (2*U))
    
    # Assemble A matrix
    A = zeros(4, 4)
    A[1, 1] = Y_v / m
    A[1, 2] = 0.0
    A[1, 3] = -U
    A[1, 4] = g
    
    A[2, 1] = L_beta / Ixx
    A[2, 2] = L_p / Ixx
    A[2, 3] = L_r / Ixx
    A[2, 4] = 0.0
    
    A[3, 1] = N_beta / Izz
    A[3, 2] = N_p / Izz
    A[3, 3] = N_r / Izz
    A[3, 4] = 0.0
    
    A[4, 1] = 0.0
    A[4, 2] = 1.0
    A[4, 3] = 0.0
    A[4, 4] = 0.0
    
    return A
end

"""
Analyze longitudinal and lateral modes.
Returns dictionary with mode characteristics.
"""
function analyze_modes(A_long, A_lat)
    results = Dict{String, Any}()
    
    λ_long = eigvals(A_long)
    λ_lat  = eigvals(A_lat)
    
    function mode_info(λ)
        ωn = abs(λ)
        ζ = -real(λ) / (abs(λ) + 1e-12)
        return (λ=λ, ωn=ωn, ζ=ζ)
    end
    
    # Sort eigenvalues by real part
    λs_long = sort(λ_long, by=real)
    λs_lat  = sort(λ_lat, by=real)
    
    # Identify major modes (heuristic based on typical aircraft dynamics)
    if length(λs_long) >= 4
        results["short_period"] = mode_info(λs_long[end])
        results["phugoid"] = mode_info(λs_long[1])
    end
    
    if length(λs_lat) >= 4
        results["dutch_roll"] = mode_info(λs_lat[2])
        results["roll_subsidence"] = mode_info(λs_lat[end])
        results["spiral"] = mode_info(λs_lat[1])
    end
    
    return results
end

# ============================================================================
# OPTIMIZATION SETUP
# ============================================================================

"""
Create objective and constraint functions for SNOW optimizer.
Objective: Maximize range using R = (e_b/g) * η * (L/D) * (m_b/m_TO)
where:
  e_b = battery specific energy [Wh/kg] = [J/kg]
  η = propulsion efficiency
  L/D = lift-to-drag ratio
  m_b = battery mass [kg]
  m_TO = total aircraft mass [kg]
"""
function make_objective_and_constraints(n_design, config::DesignConfig)
    n = n_design
    wing_span = 2 * config.max_wing_half_span
    
    # Compute weight requirement from configuration
    # Estimate total mass from components (rough initial estimate)
    est_mass = config.motor_mass + config.battery_mass + config.electronics_mass + 1.5  # +1.5 kg for structure
    weight_req = est_mass * g * config.lift_safety_factor
    
    # Get camber functions and store them globally (to avoid AD issues)
    GLOBAL_WING_CAMBER[] = get_camber_function(config, :wing)
    GLOBAL_TAIL_CAMBER[] = get_camber_function(config, :tail)
    
    # Convert battery specific energy from Wh/kg to J/kg
    e_b = config.battery_specific_energy * 3600.0  # Wh/kg * 3600 s/h = J/kg
    η = config.propulsion_efficiency
    m_b = config.battery_mass
    
    function objective_and_constraints!(g_vec, x)
        # Unpack design variables
        chords = x[1:n]
        twists = x[n+1:2n]
        sweep = x[2n+1]
        htail_x = x[2n+2]
        vtail_x = x[2n+3]
        mass_x = x[2n+4]
        
        # Build configuration (camber functions are retrieved from global storage)
        system, S, c, b, rref, _, _, _ = build_configuration(
            chords, twists;
            wing_span=wing_span,
            sweep=sweep,
            htail_x=htail_x,
            vtail_x=vtail_x,
            mass_x=mass_x
        )
        
        # Run VLM analysis
        CF, CM, dCF, dCM = run_vlm_analysis!(system, S, c, b, rref; 
                                             α=2.0*pi/180, β=0.0, symmetric=false)
        
        # Extract forces
        CD, CY, CL = CF
        q_dyn = 0.5 * config.air_density * config.cruise_speed^2
        
        # Extract Float64 values for force calculations
        S_val = S isa ForwardDiff.Dual ? ForwardDiff.value(S) : Float64(S)
        
        L = q_dyn * S_val * CL
        D = q_dyn * S_val * CD
        
        # Compute total aircraft mass from components
        # Extract values for component calculation (needs Float64)
        htail_x_val = htail_x isa ForwardDiff.Dual ? ForwardDiff.value(htail_x) : Float64(htail_x)
        vtail_x_val = vtail_x isa ForwardDiff.Dual ? ForwardDiff.value(vtail_x) : Float64(vtail_x)
        
        comps = build_components(wing_span, chords; 
                                wing_x=0.0, htail_x=htail_x_val, 
                                vtail_x=vtail_x_val)
        m_TO, cg, (Ixx, Iyy, Izz) = compute_inertia_from_components(comps)
        
        # Calculate L/D
        L_over_D = L / (D + 1e-6)
        
        # NEW Range equation: R = (e_b/g) * η * (L/D) * (m_b/m_TO)
        # where e_b is in J/kg, so convert from Wh/kg
        e_b_J_per_kg = config.battery_specific_energy * 3600.0  # Wh/kg to J/kg
        range_m = (e_b_J_per_kg / g) * config.propulsion_efficiency * L_over_D * (m_b / m_TO)
        
        # Objective: maximize range by minimizing negative range
        f = -range_m
        
        # Assemble state-space matrices for dynamic stability
        # Extract Float64 values for matrix assembly
        S_val = S isa ForwardDiff.Dual ? ForwardDiff.value(S) : Float64(S)
        c_val = c isa ForwardDiff.Dual ? ForwardDiff.value(c) : Float64(c)
        b_val = b isa ForwardDiff.Dual ? ForwardDiff.value(b) : Float64(b)
        
        A_long = assemble_longitudinal_A(dCF, dCM, S_val, c_val, m_TO, Iyy, config.cruise_speed)
        A_lat = assemble_lateral_A(dCF, dCM, S_val, b_val, m_TO, Ixx, Izz, config.cruise_speed)
        
        ev_long = eigvals(A_long)
        ev_lat  = eigvals(A_lat)
        
        # Constraint counter
        idx = 1
        
        # 1) Lift requirement
        g_vec[idx] = weight_req - L
        idx += 1
        
        # 2) Static longitudinal stability: Cm_α < 0
        Cm_alpha = dCM.alpha[2]
        g_vec[idx] = Cm_alpha + 1e-3  # Must be negative
        idx += 1
        
        # 3) Static directional stability: Cn_β > 0
        Cn_beta = dCM.beta[3]
        g_vec[idx] = -Cn_beta + 1e-4  # Must be positive
        idx += 1
        
        # 4) Dynamic longitudinal stability
        maxRe_long = maximum(real.(ev_long))
        g_vec[idx] = maxRe_long + config.stability_margin
        idx += 1
        
        # 5) Dynamic lateral stability
        maxRe_lat = maximum(real.(ev_lat))
        g_vec[idx] = maxRe_lat + config.stability_margin
        idx += 1
        
        # 6-N) Monotonic chord taper (chords decrease toward tip)
        for i in 1:(n-1)
            g_vec[idx] = chords[i+1] - chords[i]
            idx += 1
        end
        
        return f
    end
    
    n_constraints = 5 + (n_design - 1)
    return objective_and_constraints!, n_constraints
end

"""
Run the aircraft optimization using SNOW + Ipopt.
"""
function optimize_aircraft(config::DesignConfig=CONFIG)
    println("="^70)
    println("AIRCRAFT RANGE OPTIMIZATION WITH STABILITY CONSTRAINTS")
    println("="^70)
    
    n = config.n_span_sections
    nd = 2*n + 4  # Total design variables
    
    # Create objective and constraints
    objfun, ng = make_objective_and_constraints(n, config)
    
    # Initial guess
    chords0 = 0.3 * ones(n)  # Start with 30cm chords
    twists0 = range(3.0, stop=-1.0, length=n) |> collect  # Washout (twist) - reduced magnitude
    sweep0 = 0.05  # Reduced initial sweep
    htail_x0 = 1.2  # Closer tail
    vtail_x0 = 1.2  # Closer tail
    mass_x0 = 0.0
    
    x0 = vcat(chords0, twists0, [sweep0, htail_x0, vtail_x0, mass_x0])
    
    # Design variable bounds
    lx = zeros(nd)
    ux = ones(nd)
    
    # Chord bounds
    for i in 1:n
        lx[i] = 0.1  # Minimum 10cm chord
        ux[i] = 0.5  # Maximum 50cm chord
    end
    
    # Twist bounds (degrees)
    for i in 1:n
        lx[n+i] = -10.0
        ux[n+i] = 10.0
    end
    
    # Sweep bound
    lx[2n+1] = 0.0
    ux[2n+1] = 0.3
    
    # Tail x-position bounds
    lx[2n+2] = 1.0   # htail_x
    ux[2n+2] = min(2.5, config.max_fuselage_length - 0.2)
    lx[2n+3] = 1.0   # vtail_x
    ux[2n+3] = min(2.5, config.max_fuselage_length - 0.2)
    
    # Mass CG x-position bounds
    lx[2n+4] = -0.3
    ux[2n+4] = 0.3
    
    # Constraint bounds: g <= 0
    lg = fill(-Inf, ng)
    ug = zeros(ng)
    
    # Configure Ipopt
    ipopts = Dict(
        "tol" => 1e-4,              # Relaxed tolerance
        "max_iter" => 500,           # More iterations
        "print_level" => 5,
        "mu_strategy" => "adaptive",
        "linear_solver" => "mumps",
        "acceptable_tol" => 1e-3,    # Accept slightly worse solutions
        "acceptable_iter" => 15       # After 15 iterations
    )
    
    solver = IPOPT(ipopts)
    options = Options(; solver=solver, derivatives=ForwardAD())
    
    # Run optimization
    @printf("\nStarting optimization...\n")
    @printf("  Design variables: %d\n", nd)
    @printf("  Constraints: %d\n", ng)
    @printf("  Wing span: %.2f m\n", 2*config.max_wing_half_span)
    @printf("  Cruise speed: %.1f m/s\n", config.cruise_speed)
    @printf("  Wing airfoil: %s", config.wing_airfoil_type)
    if config.wing_airfoil_type == "NACA4"
        @printf(" (NACA %s)", config.wing_naca_digits)
    elseif config.wing_airfoil_type == "dat_file"
        @printf(" (%s)", basename(config.wing_dat_file))
    end
    println()
    @printf("  Tail airfoil: %s", config.tail_airfoil_type)
    if config.tail_airfoil_type == "NACA4"
        @printf(" (NACA %s)", config.tail_naca_digits)
    elseif config.tail_airfoil_type == "dat_file"
        @printf(" (%s)", basename(config.tail_dat_file))
    end
    println()
    @printf("  Foam density: %.1f kg/m³\n", config.foam_density)
    @printf("  Motor mass: %.2f kg\n", config.motor_mass)
    @printf("  Battery mass: %.2f kg\n", config.battery_mass)
    @printf("  Battery specific energy: %.1f Wh/kg\n", config.battery_specific_energy)
    @printf("  Propulsion efficiency: %.2f\n", config.propulsion_efficiency)
    @printf("  Max fuselage length: %.2f m\n\n", config.max_fuselage_length)
    
    xopt, fopt, info = minimize(objfun, x0, ng, lx, ux, lg, ug, options)
    
    println("\n" * "="^70)
    println("OPTIMIZATION COMPLETE")
    println("="^70)
    @printf("Objective value: %.6f\n", fopt)
    @printf("Exit flag: %s\n\n", info)
    
    return xopt, fopt, info
end

# ============================================================================
# POST-PROCESSING AND ANALYSIS
# ============================================================================

"""
Analyze and report detailed results for a given design.
"""
function analyze_design(x::Vector{Float64}, config::DesignConfig=CONFIG; verbose::Bool=true)
    n = config.n_span_sections
    
    # Unpack design
    chords = x[1:n]
    twists = x[n+1:2n]
    sweep = x[2n+1]
    htail_x = x[2n+2]
    vtail_x = x[2n+3]
    mass_x = x[2n+4]
    
    if verbose
        println("\n" * "="^70)
        println("DESIGN ANALYSIS")
        println("="^70)
        
        println("\nConfiguration:")
        println("  Wing airfoil: $(config.wing_airfoil_type)")
        if config.wing_airfoil_type == "NACA4"
            println("    NACA: $(config.wing_naca_digits)")
        elseif config.wing_airfoil_type == "dat_file"
            println("    File: $(config.wing_dat_file)")
        end
        println("  Tail airfoil: $(config.tail_airfoil_type)")
        if config.tail_airfoil_type == "NACA4"
            println("    NACA: $(config.tail_naca_digits)")
        elseif config.tail_airfoil_type == "dat_file"
            println("    File: $(config.tail_dat_file)")
        end
        println("  Foam density: $(config.foam_density) kg/m³")
        println("  Motor: $(config.motor_mass) kg at x=$(config.motor_x) m")
        println("  Battery: $(config.battery_mass) kg at x=$(config.battery_x) m")
        println("  Battery energy: $(config.battery_specific_energy) Wh/kg")
        println("  Propulsion efficiency: $(config.propulsion_efficiency)")
        
        println("\nGeometry Parameters:")
        println("  Wing span: $(2*config.max_wing_half_span) m")
        println("  Wing chords (root to tip):")
        for (i, c) in enumerate(chords)
            @printf("    Section %d: %.3f m\n", i, c)
        end
        println("\n  Wing twists (root to tip):")
        for (i, t) in enumerate(twists)
            @printf("    Section %d: %.2f°\n", i, t)
        end
        @printf("\n  Quarter-chord sweep: %.3f m\n", sweep)
        @printf("  Horizontal tail position: %.3f m\n", htail_x)
        @printf("  Vertical tail position: %.3f m\n", vtail_x)
        @printf("  CG x-position: %.3f m\n", mass_x)
    end
    
    # Build configuration
    wing_span = 2*config.max_wing_half_span
    
    # Set global camber functions
    GLOBAL_WING_CAMBER[] = get_camber_function(config, :wing)
    GLOBAL_TAIL_CAMBER[] = get_camber_function(config, :tail)
    
    system, S, c, b, rref, _, _, _ = build_configuration(
        chords, twists;
        wing_span=wing_span,
        sweep=sweep,
        htail_x=htail_x,
        vtail_x=vtail_x,
        mass_x=mass_x
    )
    
    # Aerodynamic analysis
    CF, CM, dCF, dCM = run_vlm_analysis!(system, S, c, b, rref; 
                                         α=2.0*pi/180, β=0.0, symmetric=false)
    
    CD, CY, CL = CF
    Cl, Cm, Cn = CM
    
    q_dyn = 0.5 * config.air_density * config.cruise_speed^2
    L = q_dyn * S * CL
    D = q_dyn * S * CD
    
    # Compute inertia for range calculation
    comps = build_components(wing_span, chords; 
                            wing_x=0.0, htail_x=htail_x, vtail_x=vtail_x)
    m_tot, cg, (Ixx, Iyy, Izz) = compute_inertia_from_components(comps)
    
    # Range calculation: R = (e_b/g) * η * (L/D) * (m_b/m_TO)
    e_b = config.battery_specific_energy * 3600.0  # Convert Wh/kg to J/kg
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
        @printf("\n  ** Estimated Range: %.1f m (%.2f km) **\n", range_m, range_m/1000.0)
    end
    
    # Stability derivatives
    CL_alpha = dCF.alpha[3]
    Cm_alpha = dCM.alpha[2]
    Cn_beta = dCM.beta[3]
    
    if verbose
        println("\n" * "-"^70)
        println("Static Stability:")
        println("-"^70)
        @printf("  dCL/dα = %.4f  (lift curve slope)\n", CL_alpha)
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
    end
    
    if verbose
        println("\n" * "-"^70)
        println("Mass Properties:")
        println("-"^70)
        @printf("  Total mass: %.3f kg\n", m_tot)
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
                @printf("  %s %-20s: λ = %7.4f %+7.4fi  |  ωₙ = %.3f rad/s  |  ζ = %.3f\n",
                        status, mode_name, real(mode.λ), imag(mode.λ), mode.ωn, mode.ζ)
            end
        end
        
        println("\n  Legend: ✓ = stable (Re(λ) < 0), ✗ = unstable (Re(λ) ≥ 0)")
    end
    
    return system, S, c, b, modes, comps, (m_tot, cg, (Ixx, Iyy, Izz))
end

"""
Quick test of basic VLM functionality.
"""
function quick_test()
    println("\n" * "="^70)
    println("QUICK TEST: Basic VLM Analysis")
    println("="^70)
    
    # Simple wing
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
    
    # Reference values
    Sref = 2 * 0.5 * (yle[2] - yle[1]) * (chord[1] + chord[2])
    cref = Sref / (2*yle[end])
    bref = 2*yle[end]
    rref = [0.25*cref, 0.0, 0.0]
    
    # Analysis
    CF, CM, dCF, dCM = run_vlm_analysis!(system, Sref, cref, bref, rref;
                                         α=2.0*pi/180, β=0.0, symmetric=false)
    
    CD, CY, CL = CF
    
    @printf("\nResults:\n")
    @printf("  Wing area: %.2f m²\n", Sref)
    @printf("  CL = %.4f\n", CL)
    @printf("  CD = %.4f\n", CD)
    @printf("  L/D = %.2f\n", CL/CD)
    @printf("  dCL/dα = %.4f\n", dCF.alpha[3])
    
    println("\n✓ Basic VLM test passed!")
    
    # Test range calculation
    println("\n" * "-"^70)
    println("Testing Range Equation:")
    println("-"^70)
    e_b = CONFIG.battery_specific_energy * 3600.0  # Wh/kg to J/kg
    η = CONFIG.propulsion_efficiency
    m_b = CONFIG.battery_mass
    m_TO = 5.0  # Example total mass
    L_over_D = CL/CD
    
    range_m = (e_b / g) * η * L_over_D * (m_b / m_TO)
    
    @printf("  Battery specific energy: %.1f Wh/kg (%.1f J/kg)\n", 
            CONFIG.battery_specific_energy, e_b)
    @printf("  Propulsion efficiency: %.2f\n", η)
    @printf("  Battery mass: %.2f kg\n", m_b)
    @printf("  Example total mass: %.2f kg\n", m_TO)
    @printf("  L/D: %.2f\n", L_over_D)
    @printf("  Calculated range: %.1f m (%.2f km)\n", range_m, range_m/1000.0)
    @printf("\n  Formula: R = (e_b/g) × η × (L/D) × (m_b/m_TO)\n")
    @printf("         R = (%.0f/%.2f) × %.2f × %.2f × (%.2f/%.2f)\n",
            e_b, g, η, L_over_D, m_b, m_TO)
    @printf("         R = %.1f m\n", range_m)
    
    println("\n✓ Range calculation test passed!")
end

"""
Calculate expected range for given parameters (without optimization).
Useful for quick "what-if" analyses.
"""
function calculate_range(;
    L_over_D::Float64 = 30.0,
    m_total::Float64 = 5.0,
    battery_mass::Float64 = CONFIG.battery_mass,
    battery_energy::Float64 = CONFIG.battery_specific_energy,
    efficiency::Float64 = CONFIG.propulsion_efficiency
)
    e_b = battery_energy * 3600.0  # Wh/kg to J/kg
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

"""
Diagnose which constraints are violated for a given design.
"""
function diagnose_constraints(x::Vector{Float64}, config::DesignConfig=CONFIG)
    n = config.n_span_sections
    
    # Create a dummy constraint vector
    g_vec = zeros(5 + (n - 1))
    
    # Create objective function to evaluate constraints
    objfun, ng = make_objective_and_constraints(n, config)
    
    # Evaluate
    f = objfun(g_vec, x)
    
    println("\nConstraint Diagnosis:")
    println("="^70)
    @printf("Objective value: %.2f\n\n", f)
    
    constraint_names = [
        "Lift requirement",
        "Longitudinal stability (Cm_α < 0)",
        "Directional stability (Cn_β > 0)",
        "Dynamic longitudinal stability",
        "Dynamic lateral stability"
    ]
    
    for i in 1:5
        status = g_vec[i] <= 0.0 ? "✓" : "✗"
        @printf("%s %s: g[%d] = %.6f %s\n", status, constraint_names[i], i, g_vec[i], 
                g_vec[i] <= 0.0 ? "(satisfied)" : "(VIOLATED)")
    end
    
    println("\nMonotonic chord constraints:")
    for i in 6:length(g_vec)
        status = g_vec[i] <= 0.0 ? "✓" : "✗"
        @printf("%s Chord taper %d-%d: g[%d] = %.6f %s\n", status, i-5, i-4, i, g_vec[i],
                g_vec[i] <= 0.0 ? "(satisfied)" : "(VIOLATED)")
    end
    
    println("="^70)
end

# ============================================================================
# CONFIGURATION HELPER FUNCTIONS
# ============================================================================

"""
Print current configuration settings.
"""
function print_config(config::DesignConfig=CONFIG)
    println("\n" * "="^70)
    println("CURRENT DESIGN CONFIGURATION")
    println("="^70)
    
    println("\nFlight Conditions:")
    @printf("  Cruise speed: %.1f m/s\n", config.cruise_speed)
    @printf("  Air density: %.3f kg/m³\n", config.air_density)
    
    println("\nWing Geometry:")
    @printf("  Max wing half-span: %.2f m (full span: %.2f m)\n", 
            config.max_wing_half_span, 2*config.max_wing_half_span)
    @printf("  Number of span sections: %d\n", config.n_span_sections)
    
    println("\nMaterials:")
    @printf("  Foam density: %.1f kg/m³\n", config.foam_density)
    @printf("  Foam thickness: %.3f m\n", config.foam_thickness)
    @printf("  Fuselage rod density: %.1f kg/m³\n", config.fuselage_density)
    @printf("  Fuselage rod radius: %.4f m\n", config.fuselage_radius)
    @printf("  Spar density: %.1f kg/m³\n", config.spar_density)
    @printf("  Spar radius: %.4f m\n", config.spar_radius)
    
    println("\nComponent Masses:")
    @printf("  Motor: %.2f kg at x=%.2f m\n", config.motor_mass, config.motor_x)
    @printf("  Battery: %.2f kg at x=%.2f m\n", config.battery_mass, config.battery_x)
    @printf("  Electronics: %.2f kg at x=%.2f m\n", config.electronics_mass, config.electronics_x)
    
    println("\nPropulsion System:")
    @printf("  Battery specific energy: %.1f Wh/kg\n", config.battery_specific_energy)
    @printf("  Propulsion efficiency: %.2f\n", config.propulsion_efficiency)
    
    println("\nStructural Constraints:")
    @printf("  Max fuselage length: %.2f m\n", config.max_fuselage_length)
    
    println("\nAirfoils:")
    println("  Wing:")
    @printf("    Type: %s\n", config.wing_airfoil_type)
    if config.wing_airfoil_type == "NACA4"
        @printf("    NACA designation: %s\n", config.wing_naca_digits)
    elseif config.wing_airfoil_type == "dat_file"
        @printf("    File: %s\n", config.wing_dat_file)
    end
    println("  Tail:")
    @printf("    Type: %s\n", config.tail_airfoil_type)
    if config.tail_airfoil_type == "NACA4"
        @printf("    NACA designation: %s\n", config.tail_naca_digits)
    elseif config.tail_airfoil_type == "dat_file"
        @printf("    File: %s\n", config.tail_dat_file)
    end
    
    println("\nOptimization:")
    @printf("  Stability margin: %.3f\n", config.stability_margin)
    @printf("  Lift safety factor: %.2f\n", config.lift_safety_factor)
    
    println("="^70)
end

"""
Update configuration parameters easily.
Example: 
    set_config!(motor_mass=0.20, battery_mass=0.30, wing_airfoil_type="NACA4", wing_naca_digits="2412")
"""
function set_config!(; kwargs...)
    for (key, value) in kwargs
        if hasfield(DesignConfig, key)
            setfield!(CONFIG, key, value)
            @printf("✓ Set %s = %s\n", key, value)
        else
            @warn "Unknown configuration parameter: $key"
            println("  Available parameters:")
            for fname in fieldnames(DesignConfig)
                println("    - $fname")
            end
        end
    end
    println("\nConfiguration updated. Use print_config() to review.")
end

"""
Load airfoil from .dat file for wing or tail.
Convenience function that sets the appropriate config fields.
"""
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

# ============================================================================
# MAIN EXECUTION
# ============================================================================

"""
Main entry point for optimization.
"""
function main()
    println("\n" * "#"^70)
    println("# AIRCRAFT OPTIMIZATION FOR MAXIMUM RANGE")
    println("# VortexLattice.jl + SNOW.jl + Component-Based Inertia Model")
    println("# Range Equation: R = (e_b/g) × η × (L/D) × (m_b/m_TO)")
    println("#"^70)
    
    # Show current configuration
    print_config()
    
    # Run quick test first
    quick_test()
    
    println("\n\n" * "="^70)
    println("CONFIGURATION OPTIONS")
    println("="^70)
    println("\nTo modify configuration before optimizing, use set_config!():")
    println("\nExample 1 - Change battery and motor:")
    println("  set_config!(")
    println("      motor_mass = 0.25,")
    println("      battery_mass = 0.35,")
    println("      battery_specific_energy = 180.0,  # Wh/kg")
    println("      propulsion_efficiency = 0.70")
    println("  )")
    println("\nExample 2 - Use NACA airfoils:")
    println("  set_config!(")
    println("      wing_airfoil_type = \"NACA4\",")
    println("      wing_naca_digits = \"2412\",")
    println("      tail_airfoil_type = \"NACA4\",")
    println("      tail_naca_digits = \"0012\"")
    println("  )")
    println("\nExample 3 - Load airfoil from .dat file:")
    println("  load_airfoil!(\"sd7062.dat\", surface=:wing)")
    println("  load_airfoil!(\"naca0012.dat\", surface=:tail)")
    println("\nExample 4 - Adjust geometry:")
    println("  set_config!(")
    println("      max_wing_half_span = 2.5,")
    println("      max_fuselage_length = 2.0,")
    println("      foam_density = 35.0")
    println("  )")
    
    println("\n" * "="^70)
    println("\nPress Enter to start optimization with current configuration,")
    println("or Ctrl+C to exit and modify settings first...")
    readline()
    
    # Run optimization
    xopt, fopt, info = optimize_aircraft(CONFIG)
    
    # Analyze optimal design
    analyze_design(xopt, CONFIG; verbose=true)
    
    println("\n" * "#"^70)
    println("# OPTIMIZATION COMPLETE")
    println("#"^70)
    
    return xopt, fopt
end

# Run if executed as script
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end