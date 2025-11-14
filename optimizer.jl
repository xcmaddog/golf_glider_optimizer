# GliderOptimizer.jl
# Template optimizer using VortexLattice.jl + SNOW.jl + IPOPT
# - Parametrizes wing chords & twists (spanwise sections)
# - Adds simple horizontal & vertical tails at tunable x-locations
# - Places a point mass (x position) for CG movement
# - Computes stability derivatives with central finite differences
# - Builds linearized A-matrices and enforces dynamic stability (Re(eig)< -margin)
# - Enforces monotonic chord (root -> tip thinner)
#
# NOTE: This is a template. Tune FD steps, tail sizing, and solver options for production.

using VortexLattice
using SNOW
using LinearAlgebra
using Printf
using Plots

# ----------------------------
# Physical / reference values
# ----------------------------
const ρ = 1.225
const g = 9.80665
const U0 = 10.0               # cruise speed [m/s] - change to desired
const mass_total = 5.0        # total mass [kg] (including added point mass) - adjust
const weight = mass_total * g
const n_span_sections = 9     # number of chord control points (design dim)
const wing_half_span = 2.0    # half span of wing [m] (so total span = 4 m)
const default_tail_arm = 1.0  # nominal tail arm if not optimized

# ---------- helper: build full configuration ----------
# We reuse the idea of build_wing from your file but add tail surfaces and CG placement.
# For simplicity the fuselage is a point-mass along x-axis, with CG at x_cg.
# We create:
#  - wing (symmetric) built from chord vector + twist vector
#  - horizontal tail (symm) as small rectangular surface positioned at (x_ht, 0, z_ht)
#  - vertical tail (unsymmetric) as single surface (we place it on fuselage centerline)
#
# Return: surfaces vector, reference_area (wing + tails), reference_chord, grid (wing grid), rref (ref point used)
function build_configuration(chords::Vector{Float64}, twists::Vector{Float64};
                             wing_span = 2*wing_half_span,
                             sweep = 0.0,     # sweep of wing quarter-chord in meters (positive aft)
                             htail_x = 1.0,   # horizontal tail x-location (aft of wing quarter-chord reference)
                             vtail_x = 1.0,
                             mass_x = 0.0,    # x position of point mass (cg)
                             htail_area = 0.3, vtail_area = 0.1,
                             ht_span = 0.6, vt_height = 0.6)
    # wing quarter-chord leading-edge x positions (align quarter chords at x = -sweep/4 ???)
    n = length(chords)
    # build wing quarter-chord leading edge arrays for half-wing (we rely on symmetric option)
    xle = fill(-0.25 .* chords, 1) # placeholder: we will adjust positions to include sweep
    # Actually, xle should be spanwise vector; generate arrays of length n
    xle = chords .* (-1/4) .+ sweep .* collect(range(0, stop=1, length=n)) # linear sweep approximation
    yle = collect(range(0.0, stop=wing_span/2, length=n))
    zle = zeros(n)
    theta = (pi/180) .* twists
    phi = zeros(n)
    ns = n - 1
    nc = 9
    wing_grid, wing_surface = wing_to_surface_panels(xle, yle, zle, chords, theta, phi, ns, nc)
    # compute wing area and ref chord (approx numeric trapezoid)
    wing_area = 0.0
    for i in 1:(n-1)
        y0 = yle[i]; y1 = yle[i+1]
        wing_area += 2 * 0.5 * (y1 - y0) * (chords[i] + chords[i+1]) # symmetric both halves
    end

    # horizontal tail (symmetric)
    # simple rectangular tail: chord_ht centered at quarter chord aligned
    ht_chord = htail_area / (ht_span) # area = chord * span*2 (since symmetric) -> chord = area/(2*ht_span) but we'll use ht_span as half-span
    # We'll build as n_ht sections = 3
    n_ht = 3
    ht_chords = fill(ht_chord, n_ht)
    ht_xle = fill(htail_x .- 0.25*ht_chords[1], n_ht)  # quarter-chord alignment at htail_x
    ht_yle = collect(range(0.05, stop=ht_span, length=n_ht))
    ht_zle = fill(0.0, n_ht) .+ 0.0 # same plane as fuselage
    ht_theta = zeros(n_ht)
    ht_phi = zeros(n_ht)
    ht_ns = n_ht - 1
    ht_nc = 5
    _, htail_surface = wing_to_surface_panels(ht_xle, ht_yle, ht_zle, ht_chords, ht_theta, ht_phi, ht_ns, ht_nc)

    # vertical tail - we'll create an unsymmetric surface oriented vertically
    # to use wing_to_surface_panels we create a wing rotated: treat vertical tail as wing in z-direction
    vt_chord = vtail_area / vt_height
    n_vt = 3
    vt_chords = fill(vt_chord, n_vt)
    # For vertical tail we set y coordinates near root (centerline), and z coordinates as spanwise coordinate
    vt_xle = fill(vtail_x .- 0.25*vt_chords[1], n_vt)
    vt_yle = fill(0.0, n_vt)
    vt_zle = collect(range(0.0, stop=vt_height, length=n_vt))
    vt_theta = zeros(n_vt)
    vt_phi = zeros(n_vt)
    vt_ns = n_vt - 1
    vt_nc = 5
    # wing_to_surface_panels expects y as lateral coord; but to approximate vertical tail we will build as a wing and then rotate coordinates later
    grid_vt, vt_surface = wing_to_surface_panels(vt_xle, vt_yle, vt_zle, vt_chords, vt_theta, vt_phi, vt_ns, vt_nc)
    # Now rotate the vertical tail surface panels so that 'span' goes in z-direction: VortexLattice Surface likely stores panel corner coordinates; we hack by translating coordinates:
    # (Simplification) Instead, we will approximate vertical tail by a small asymmetric surface placed at centerline with sideslip responses captured approximately.
    # many VLM libraries support creating arbitrary surfaces; for this template we keep vt_surface as-is but note it's an approximation.

    # Combine surfaces
    surfaces = [wing_surface, htail_surface, vt_surface]
    # reference area (wing only or include tails? We'll use wing area for reference but include tails for total area)
    total_area = wing_area + htail_area + vtail_area
    reference_chord = wing_area / (wing_span)  # approximate mean chord
    # reference location rref (set to cg)
    rref = [mass_x, 0.0, 0.0]
    return surfaces, total_area, reference_chord, wing_grid, rref
end

# ----------------------------
# Run steady analysis wrapper
# ----------------------------
function run_vlm_analysis(surfaces, ref_area, ref_chord, rref; α = 0.0, β = 0.0)
    Vinf = U0
    fs = Freestream(Vinf, α, β, [0.0, 0.0, 0.0])
    # reference uses: area, chord, panels (choose 8), rref, Vinf
    ref = Reference(ref_area, ref_chord, n_span_sections-1, rref, Vinf)
    sys = steady_analysis(surfaces, ref, fs; symmetric = true)
    CF, CM = body_forces(sys; frame = Wind())
    CD, CY, CL = CF
    # Return forces and moments in body/ wind frame:
    # CF = (CD, CY, CL) ; CM = (Cl_mom, Cm_mom, Cn_mom)
    properties = get_surface_properties(sys)
    return CL, CD, CY, CM, properties, sys
end

# ----------------------------
# Finite-difference stability derivatives
# ----------------------------
# We compute central differences for:
# Longitudinal: CL_alpha, CD_alpha, Cm_alpha, M_q (pitch rate), Z_q, etc.
# Lateral: Y_beta, N_beta, L_p, N_r, etc.
#
# For generality we'll perturb small α (angle of attack), q (pitch rate), β (sideslip),
# p/r roll/yaw rates are more complex, but we can approximate by small rotational freestream (noting VortexLattice steady_analysis expects Omega in Freestream)
#
function compute_stability_derivatives(design_surfaces, ref_area, ref_chord, rref; 
                                       dα = 0.5 * (pi/180),      # 0.5 deg
                                       dq = 0.01,               # small non-dimensional rotation? we will perturb angular rates in rad/s
                                       dβ = 0.5 * (pi/180))
    # baseline
    CL0, CD0, CY0, CM0, props0, sys0 = run_vlm_analysis(design_surfaces, ref_area, ref_chord, rref; α=0.0, β=0.0)
    # perturb alpha (+ and -)
    CL_p, CD_p, CY_p, CM_p, _, _ = run_vlm_analysis(design_surfaces, ref_area, ref_chord, rref; α=dα, β=0.0)
    CL_m, CD_m, CY_m, CM_m, _, _ = run_vlm_analysis(design_surfaces, ref_area, ref_chord, rref; α=-dα, β=0.0)
    # derivatives
    CL_alpha = (CL_p - CL_m) / (2dα)
    CD_alpha = (CD_p - CD_m) / (2dα)
    Cm_alpha = (CM_p[2] - CM_m[2]) / (2dα)  # pitching moment derivative about y-axis (index 2 usually)
    # sideslip derivatives
    CL_b_p, CD_b_p, CY_b_p, CM_b_p, _, _ = run_vlm_analysis(design_surfaces, ref_area, ref_chord, rref; α=0.0, β=dβ)
    CL_b_m, CD_b_m, CY_b_m, CM_b_m, _, _ = run_vlm_analysis(design_surfaces, ref_area, ref_chord, rref; α=0.0, β=-dβ)
    CY_beta = (CY_b_p - CY_b_m) / (2dβ)
    N_beta = (CM_b_p[3] - CM_b_m[3]) / (2dβ) # yawing moment derivative (Cn_beta) approx
    # Pitch-rate derivatives: approximate by running with small freestream rotation Omega about pitch axis (y)
    # VortexLattice's Freestream accepts Omega vector (Ω_x, Ω_y, Ω_z)
    fs_p = Freestream(U0, 0.0, 0.0, [0.0, dq, 0.0])
    ref = Reference(ref_area, ref_chord, 8, rref, U0)
    sys_qp = steady_analysis(design_surfaces, ref, fs_p; symmetric = true)
    CF_qp, CM_qp = body_forces(sys_qp; frame = Wind())
    # negative rotation
    fs_qm = Freestream(U0, 0.0, 0.0, [0.0, -dq, 0.0])
    sys_qm = steady_analysis(design_surfaces, ref, fs_qm; symmetric = true)
    CF_qm, CM_qm = body_forces(sys_qm; frame = Wind())
    M_q = (CM_qp[2] - CM_qm[2]) / (2dq)
    # roll & yaw rates: use Omega perturbation about x and z
    fs_pp = Freestream(U0, 0.0, 0.0, [dq, 0.0, 0.0])
    sys_pp = steady_analysis(design_surfaces, ref, fs_pp; symmetric = true)
    CF_pp, CM_pp = body_forces(sys_pp; frame = Wind())
    L_p = (CM_pp[1] - CM_qp[1]) / (dq)  # crude approx: compare to previous; this is heuristic
    # yaw rate
    fs_rp = Freestream(U0, 0.0, 0.0, [0.0, 0.0, dq])
    sys_rp = steady_analysis(design_surfaces, ref, fs_rp; symmetric = true)
    CF_rp, CM_rp = body_forces(sys_rp; frame = Wind())
    N_r = (CM_rp[3] - CM_qp[3]) / dq

    # package derivatives into a dictionary for later assembly
    derivs = Dict(
        :CL_alpha => CL_alpha,
        :CD_alpha => CD_alpha,
        :Cm_alpha => Cm_alpha,
        :M_q => M_q,
        :CY_beta => CY_beta,
        :N_beta => N_beta,
        :L_p => L_p,
        :N_r => N_r,
        :CL0 => CL0,
        :CD0 => CD0,
        :CY0 => CY0,
        :CM0 => CM0
    )

    return derivs
end

# ----------------------------
# Build linearized A-matrices
# ----------------------------
# We build simplified 4x4 longitudinal and 4x4 lateral matrices suitable for eigenvalue extraction.
# The expressions used here are simplified (assume small angle, neglect Z_dot_w etc).
# For more accurate modelling include added mass terms and correct sign conventions.
function build_longitudinal_A(derivs, mass, Iyy, ref_chord, U)
    # State: [u, w, q, θ]
    # We need X_u, X_w etc. But with limited derivative set, we will form an approximate A.
    # Use classic small-state simplification (see Stevens & Lewis) with:
    # u_dot ≈ X_u/m * u + X_w/m * w - g * θ
    # w_dot ≈ Z_u/m * u + Z_w/m * w + U * q + ...
    # q_dot ≈ M_u/Iyy * u + M_w/Iyy * w + M_q/Iyy * q
    # θ_dot = q
    #
    # We don't have exact X_, Z_, etc. We'll use derivatives with respect to α:
    CLα = derivs[:CL_alpha]
    Cmα = derivs[:Cm_alpha]
    # Convert to dimensional derivatives: e.g., L = 0.5 rho U^2 S * CL
    qdyn = 0.5 * ρ * U^2
    Sref = ref_chord * (2*wing_half_span) # approximate wing area
    # approximate Z_w (vertical force w.r.t. w) from CL_alpha: dL/dα * dα/dw
    # α ≈ w/U => dα/dw = 1/U
    Z_w = - qdyn * Sref * CLα * (1/U)    # negative because Z is positive downward in many sign conventions; we pick consistent signs crudely
    X_w = 0.0
    M_w = qdyn * ref_chord * Sref * Cmα * (1/U)
    M_q = derivs[:M_q] * qdyn * ref_chord * Sref  # scale M_q (non-dimensional -> dimensional) heuristic
    # assemble A
    A = zeros(4,4)
    A[1,1] = 0.0 # X_u/m unknown
    A[1,2] = X_w / mass
    A[1,3] = 0.0
    A[1,4] = -g

    A[2,1] = 0.0
    A[2,2] = Z_w / mass
    A[2,3] = U + 0.0
    A[2,4] = 0.0

    A[3,1] = 0.0
    A[3,2] = M_w / Iyy
    A[3,3] = M_q / Iyy
    A[3,4] = 0.0

    A[4,1] = 0.0
    A[4,2] = 0.0
    A[4,3] = 1.0
    A[4,4] = 0.0

    return A
end

function build_lateral_A(derivs, mass, Ixx, Izz, ref_chord, U)
    # state: [v, p, r, φ]
    # simplified approximate coefficients:
    qdyn = 0.5 * ρ * U^2
    Sref = ref_chord * (2*wing_half_span)
    Y_v = qdyn * Sref * (-derivs[:CY_beta]) * (1.0 / U)  # approximate
    N_beta = qdyn * ref_chord * Sref * derivs[:N_beta]
    L_p = derivs[:L_p]
    N_r = derivs[:N_r]

    A = zeros(4,4)
    A[1,1] = Y_v / mass
    A[1,2] = 0.0
    A[1,3] = (1.0 / mass) * 0.0  # approx
    A[1,4] = g

    A[2,1] = 0.0
    A[2,2] = L_p / Ixx
    A[2,3] = 0.0
    A[2,4] = 0.0

    A[3,1] = N_beta / Izz
    A[3,2] = 0.0
    A[3,3] = N_r / Izz
    A[3,4] = 0.0

    A[4,1] = 0.0
    A[4,2] = 1.0
    A[4,3] = tan(0.0) # for small trim
    A[4,4] = 0.0

    return A
end

# ----------------------------
# Objective & constraints for SNOW
# ----------------------------
# Decision vector x layout:
# [chords[1:n], twists[1:n], sweep, htail_x, vtail_x, mass_x]
function make_objective_and_constraints(n_design;
                                        wing_span = 2*wing_half_span,
                                        lift_requirement = weight,
                                        mass_total = mass_total,
                                        static_margin_req = 0.0,
                                        dyn_margin = 0.01)
    n = n_design
    function objective_and_constraints!(gvec, x)
        # unpack
        chords = x[1:n]
        twists = x[n+1:2n]
        sweep = x[2n+1]
        htail_x = x[2n+2]
        vtail_x = x[2n+3]
        mass_x = x[2n+4]

        # build config
        surfaces, ref_area, ref_chord, wing_grid, rref = build_configuration(chords, twists;
                                                                             wing_span = wing_span,
                                                                             sweep = sweep,
                                                                             htail_x = htail_x,
                                                                             vtail_x = vtail_x,
                                                                             mass_x = mass_x)
        # run base analysis
        CL0, CD0, CY0, CM0, props0, sys0 = run_vlm_analysis(surfaces, ref_area, ref_chord, rref; α=0.0, β=0.0)

        # compute derivatives
        derivs = compute_stability_derivatives(surfaces, ref_area, ref_chord, rref)

        # objective: maximize L/D -> minimize -L/D
        # dimensional lift & drag:
        qdyn = 0.5 * ρ * U0^2
        L_dim = qdyn * ref_area * CL0
        D_dim = qdyn * ref_area * CD0
        f = - (L_dim / (D_dim + 1e-12))  # minimize

        # constraints vector gvec: we'll produce several inequality constraints (we require g <= 0)
        # layout:
        # g1: lift constraint: L >= lift_requirement -> g1 = lift_requirement - L (<=0)
        # g2: Cm_alpha <= -eps -> g2 = Cm_alpha + eps (<=0)
        # g3: directional static N_beta >= small -> g3 = -N_beta + eps (<=0)
        # g4: max real part of longitudinal eigenvalues <= -dyn_margin -> g4 = maxRe_long + dyn_margin (<=0)
        # g5: max real part of lateral eigenvalues <= -dyn_margin -> g5 = maxRe_lat + dyn_margin (<=0)
        # g6..: monotonicity: chord[i+1] - chord[i] <= 0

        # fill gvec
        ng = length(gvec)
        # constraint 1
        gvec[1] = lift_requirement - L_dim

        # static Cm_alpha
        eps_static = 0.005
        gvec[2] = derivs[:Cm_alpha] + eps_static  # require Cm_alpha <= -eps_static

        # directional static (N_beta positive)
        eps_dir = 0.0001
        gvec[3] = - derivs[:N_beta] + eps_dir

        # dynamic constraints
        # estimate inertias (very approximate) as slender body values
        Ixx = 0.1 * mass_total  # crude
        Iyy = 0.2 * mass_total
        Izz = 0.25 * mass_total

        A_long = build_longitudinal_A(derivs, mass_total, Iyy, ref_chord, U0)
        A_lat  = build_lateral_A(derivs, mass_total, Ixx, Izz, ref_chord, U0)
        ev_long = eigvals(A_long)
        ev_lat = eigvals(A_lat)
        maxRe_long = maximum(real.(ev_long))
        maxRe_lat = maximum(real.(ev_lat))
        gvec[4] = maxRe_long + dyn_margin
        gvec[5] = maxRe_lat + dyn_margin

        # monotonicity chords (n-1 constraints)
        for i in 1:(n-1)
            gvec[5 + i] = chords[i+1] - chords[i]  # <= 0 ensures chord decreases root->tip
        end

        # optional: bound tail positions (we could also enforce via variable bounds)
        return f
    end

    return objective_and_constraints!, (1 + 1 + 1 + 1 + 1 + (n - 1))  # objective function and number of constraints
end

# ----------------------------
# Run optimizer (SNOW + IPOPT)
# ----------------------------
function optimize_plane()
    n = n_span_sections
    nd = 2n + 4  # chords + twists + sweep + h_x + v_x + mass_x
    # build objective & constraints
    objfun, ng = make_objective_and_constraints(n; lift_requirement = weight, mass_total = mass_total, dyn_margin = 0.02)
    # initial guess
    chords0 = 0.25 .* ones(n)           # 25 cm root->tip initial
    twists0 = zeros(n)                  # no twist initially
    sweep0 = 0.0
    htail_x0 = default_tail_arm
    vtail_x0 = default_tail_arm
    mass_x0 = 0.0
    x0 = vcat(chords0, twists0, [sweep0, htail_x0, vtail_x0, mass_x0])

    # bounds
    lx = zeros(nd) .+ 0.03   # min chord 3 cm, min twist -10 deg, etc
    ux = vcat(fill(1.0, n), fill(10.0, n), [1.0, 3.0, 3.0, 1.5]) # chords up to 1 m, twists up to 10 deg, sweep 1 m, tail placements reasonable, mass x within fuselage

    # more realistic variable bounds:
    for i in 1:n
        # twists bounds in deg: -10 to +10
        lx[n + i] = -15.0
        ux[n + i] = 15.0
    end

    # number of constraints
    ng_total = ng
    # lower/upper bounds on g (we require g <= 0 -> set lower = -Inf, upper = 0)
    lg = fill(-Inf, ng_total)
    ug = zeros(ng_total)

    # SNOW IPOPT options
    ipopts = Dict("tol" => 1e-6, "max_iter" => 200)
    solver = IPOPT(ipopts)
    options = Options(; solver)

    # define wrapper for SNOW minimize signature: objective!(g,x) returns objective
    function objective_for_snow!(g,x)
        return objfun(g,x)
    end

    # run minimize
    @printf("Starting optimization with %d decision variables and %d constraints\n", nd, ng_total)
    xopt, fopt, info = minimize(objective_for_snow!, x0, ng_total, lx, ux, lg, ug, options)

    @printf("Done. fopt = %f\n", fopt)
    return xopt, fopt, info
end

# ----------------------------
# Execute optimization (small example)
# ----------------------------
if abspath(PROGRAM_FILE) == @__FILE__
    xopt, fopt, info = optimize_plane()
    println("xopt = ", xopt)
    println("info = ", info)
end
