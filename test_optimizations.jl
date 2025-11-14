#!/usr/bin/env julia
# Manual test script to verify GRASP optimizations
# Run with: julia --project=. test_optimizations.jl

using Metaheuristics
using Test

println("Testing GRASP Optimizations...")
println("=" ^ 60)

# Test 1: GRASP correctness on knapsack problem
@testset "GRASP Knapsack Correctness" begin
    println("\n[Test 1] GRASP finds optimal solution on knapsack...")

    struct KPInstance
        profit
        weight
        capacity
    end

    function Metaheuristics.compute_cost(candidates, constructor, instance::KPInstance)
        ratio = instance.profit[candidates] ./ instance.weight[candidates]
        maximum(ratio) .- ratio
    end

    profit = [55, 10, 47, 5, 4, 50, 8, 61, 85, 87]
    weight = [95, 4, 60, 32, 23, 72, 80, 62, 65, 46]
    capacity = 269
    optimum = 295

    f, search_space, _ = Metaheuristics.TestProblems.knapsack(profit, weight, capacity)

    candidates = rand(search_space)
    instance = KPInstance(profit, weight, capacity)
    constructor = Metaheuristics.GreedyRandomizedConstructor(;candidates, instance, α=0.95)
    local_search = Metaheuristics.BestImproveSearch()
    grasp = GRASP(;constructor, local_search)

    options = Options(iterations=50, seed=1)
    result = optimize(f, search_space, grasp, options=options)

    @test -minimum(result) == optimum
    println("  ✓ Found optimal solution: $(-minimum(result)) == $optimum")
end

# Test 2: Convergence tracking doesn't explode memory
@testset "Convergence Tracking Memory" begin
    println("\n[Test 2] Convergence tracking memory optimization...")

    struct KPInstance
        profit
        weight
        capacity
    end

    function Metaheuristics.compute_cost(candidates, constructor, instance::KPInstance)
        ratio = instance.profit[candidates] ./ instance.weight[candidates]
        maximum(ratio) .- ratio
    end

    profit = [55, 10, 47, 5, 4, 50, 8, 61, 85, 87]
    weight = [95, 4, 60, 32, 23, 72, 80, 62, 65, 46]
    capacity = 269

    f, search_space, _ = Metaheuristics.TestProblems.knapsack(profit, weight, capacity)

    candidates = rand(search_space)
    instance = KPInstance(profit, weight, capacity)
    constructor = Metaheuristics.GreedyRandomizedConstructor(;candidates, instance, α=0.95)
    local_search = Metaheuristics.BestImproveSearch()
    grasp = GRASP(;constructor, local_search)

    options = Options(iterations=100, seed=1, store_convergence=true)
    result = optimize(f, search_space, grasp, options=options)

    # Check convergence data exists
    @test !isempty(result.convergence)
    @test length(result.convergence) > 0

    # Verify convergence states don't have populations (optimization)
    for conv_state in result.convergence
        @test isempty(conv_state.population)
        @test isempty(conv_state.convergence)  # No recursive convergence
    end

    println("  ✓ Convergence tracked without population copies")
    println("  ✓ No recursive convergence (prevented exponential memory)")
    println("  ✓ Convergence entries: $(length(result.convergence))")
end

# Test 3: FirstImproveSearch works correctly
@testset "FirstImproveSearch" begin
    println("\n[Test 3] FirstImproveSearch local search...")

    struct KPInstance
        profit
        weight
        capacity
    end

    function Metaheuristics.compute_cost(candidates, constructor, instance::KPInstance)
        ratio = instance.profit[candidates] ./ instance.weight[candidates]
        maximum(ratio) .- ratio
    end

    profit = [55, 10, 47, 5, 4, 50, 8, 61, 85, 87]
    weight = [95, 4, 60, 32, 23, 72, 80, 62, 65, 46]
    capacity = 269
    optimum = 295

    f, search_space, _ = Metaheuristics.TestProblems.knapsack(profit, weight, capacity)

    candidates = rand(search_space)
    instance = KPInstance(profit, weight, capacity)
    constructor = Metaheuristics.GreedyRandomizedConstructor(;candidates, instance, α=0.95)
    local_search = Metaheuristics.FirstImproveSearch()  # Test FirstImprove instead of BestImprove
    grasp = GRASP(;constructor, local_search)

    options = Options(iterations=50, seed=1)
    result = optimize(f, search_space, grasp, options=options)

    @test -minimum(result) == optimum
    println("  ✓ FirstImproveSearch finds optimal solution")
end

# Test 4: Constructor with different α values
@testset "Constructor α parameter" begin
    println("\n[Test 4] Constructor with different α values...")

    struct KPInstance
        profit
        weight
        capacity
    end

    function Metaheuristics.compute_cost(candidates, constructor, instance::KPInstance)
        ratio = instance.profit[candidates] ./ instance.weight[candidates]
        maximum(ratio) .- ratio
    end

    profit = [55, 10, 47, 5, 4, 50, 8, 61, 85, 87]
    weight = [95, 4, 60, 32, 23, 72, 80, 62, 65, 46]
    capacity = 269

    f, search_space, _ = Metaheuristics.TestProblems.knapsack(profit, weight, capacity)

    for α in [0.0, 0.5, 0.95, 1.0]
        candidates = rand(search_space)
        instance = KPInstance(profit, weight, capacity)
        constructor = Metaheuristics.GreedyRandomizedConstructor(;candidates, instance, α=α)
        local_search = Metaheuristics.BestImproveSearch()
        grasp = GRASP(;constructor, local_search)

        options = Options(iterations=20, seed=1)
        result = optimize(f, search_space, grasp, options=options)

        @test minimum(result) isa Number
        println("  ✓ α=$α works correctly (f=$(round(-minimum(result), digits=1)))")
    end
end

println("\n" * "=" ^ 60)
println("All optimization tests passed! ✓")
println("=" ^ 60)
