"""
    GreedyRandomizedConstructor(;candidates, instance, α, rng)

This structure can be used to use the Greedy Randomized Contructor.

- `candidates` vector of candidates (item to choose to form a solution).
- `instance` a user define structure for saving data on the problem instance.
- `α`  controls the randomness (0 deterministic greedy heuristic, 1 for pure random).
- `rng` storages the random number generator.

This constructor assumes overwriting the `compute_cost` function:

```julia
import Metaheuristics as MH
struct MyInstance end
function MH.compute_cost(candidates, constructor, instance::MyInstance)
    # ...
end
```

See also [`compute_cost`](@ref) and [`GRASP`](@ref)
"""
Base.@kwdef struct GreedyRandomizedConstructor
    candidates
    instance = nothing
    α::Float64 = 0.6
    rng = default_rng_mh()
end

"""
    compute_cost(candidates, constructor, instance)

Compute the cost for each candidate in `candidates`, for given `constructor` and
provided `instance`.

See also [`GreedyRandomizedConstructor`](@ref) and [`GRASP`](@ref)
"""
function compute_cost(candidates, constructor, instance)
    @warn "Define compute_cost for\nconstructor=$constructor\ninstance=$instance"
    zeros(length(candidates))
end

"""
    construct(constructor)

Constructor procedure for GRASP.

See also [`GreedyRandomizedConstructor`](@ref), [`compute_cost`](@ref) and [`GRASP`](@ref)
"""
function construct(constructor::GreedyRandomizedConstructor)
    # Use index array instead of copying entire candidates array
    # This avoids O(n) memory allocation and improves cache locality
    remaining_indices = collect(eachindex(constructor.candidates))
    candidates = constructor.candidates
    α = constructor.α
    # create empty solution S
    S = similar(candidates, 0)
    # construct solution
    while !isempty(remaining_indices)
        # View only the remaining candidates
        available = @view candidates[remaining_indices]
        cost = compute_cost(available, constructor, constructor.instance)
        cmin = minimum(cost)
        cmax = maximum(cost)
        # compute restricted candidate list (indices into available, not original candidates)
        RCL = [i for i in eachindex(cost) if cost[i] <= cmin + α*(cmax - cmin) ]
        if isempty(RCL)
            @error "RCL is empty. Try increasing α or check your `compute_cost` method."
            return
        end

        # select candidate at random and insert into solution
        rcl_idx = rand(constructor.rng, RCL)
        original_idx = remaining_indices[rcl_idx]
        push!(S, candidates[original_idx])
        # update list of remaining indices (much cheaper than deleteat! on full candidates)
        deleteat!(remaining_indices, rcl_idx)
    end
    S
end
