package be.brw.domain;

import be.brw.config.GAConfig;
import be.brw.domain.strategy.*;

import java.util.*;

/**
 * Implements the core logic of a genetic algorithm to solve a bitstring-matching problem.
 * <p>
 * This class orchestrates the evolutionary process, including population initialization,
 * selection, crossover, and mutation, over a series of generations. It is configured
 * via a {@link GAConfig} object, which specifies all parameters for the simulation.
 * </p>
 */
public class GeneticAlgorithm {

    public final int MAX_THEORETICAL_FITNESS = 4;

    /**
     * Configuration object containing all parameters for the genetic algorithm.
     */
    private final GAConfig config;
    /**
     * Random number generator used for all stochastic operations (selection, crossover, mutation).
     */
    private final Random random;
    /**
     * The current population of individuals. This object is replaced with a new population each generation.
     */
    private Population population;
    /**
     * The current number of generations
     */
    private int generationCount;

    /**
     * Constructs a new GeneticAlgorithm instance and initializes the first population.
     *
     * @param configuration The {@link GAConfig} object that defines the parameters of the algorithm.
     */
    public GeneticAlgorithm(GAConfig configuration){
        this.config = configuration;
        this.random = new Random(config.getSeed());

        // Initialize the starting population based on the configuration.
        this.population = new Population(
                config.getPopulationSize(),
                config.getMinGenomeLength(),
                config.getMaxGenomeLength(),
                config.getSeed(),
                config.getLengthPunishingStrategy(),
                config.getLengthPunishingFactor()
        );

        this.generationCount = 0;
    }

    /**
     * Gets the current generation count.
     *
     * @return The number of generations that have been processed.
     */
    public int getGenerationCount(){
        return this.generationCount;
    }

    /**
     * Executes the genetic algorithm for a configured number of generations.
     * <p>
     * The algorithm proceeds generation by generation, applying selection, crossover, and mutation
     * to evolve the population toward the target solution. The process terminates if a perfect
     * solution is found or the maximum number of generations is reached.
     * </p>
     * @return The fittest individual found after the algorithm completes or finds a solution.
     */
    public Individual runAlgorithm() {
        int maxGeneration = config.getMaxGeneration();
        MutationTargetStrategy mutTarget = config.getMutationTargetStrategy();
        int eliteCount = (int) Math.round(config.getPopulationSize() * (1.0 - config.getCrossoverRate()));
        for (int i = 0; i <= maxGeneration; i++){
            this.generationCount = i;

            List<Individual> individuals = this.population.getIndividuals();

            // Check for a perfect solution in the current population.
            for (Individual individual: individuals){
                if(individual.getFitness() == MAX_THEORETICAL_FITNESS){
                    System.out.println("Solution found in " + i + " generations");
                    return this.population.getFittest();
                }
            }

            // 1. Selection: Select the "elite" individuals to survive to the next generation.
            List<Individual> survivors = selection(individuals, eliteCount);

            // 2. Mutation (on parents): Optionally mutate the selected survivors.
            if (mutTarget == MutationTargetStrategy.PARENTS || mutTarget == MutationTargetStrategy.BOTH) {
                for (int j = 0; j < survivors.size(); j++) {
                    if (random.nextDouble() <= config.getMutationRate()) {
                        survivors.set(j, mutate(survivors.get(j)));
                    }
                }
            }

            // 3. Crossover and Mutation (on children): Create new children to fill the rest of the population.
            List<Individual> children = new ArrayList<>(config.getPopulationSize() - eliteCount);
            while (eliteCount + children.size() < config.getPopulationSize()) {
                List<Individual> parents = selection(survivors, 2);
                Individual child = crossover(parents.getFirst(), parents.getLast());
                if (mutTarget == MutationTargetStrategy.CHILDREN || mutTarget == MutationTargetStrategy.BOTH) {
                    if (random.nextDouble() <= config.getMutationRate()) {
                        mutate(child);
                    }
                }

                children.add(child);
            }

            // Create the next generation's population from survivors and new children.
            survivors.addAll(children);
            this.population = new Population(survivors, config.getSeed(), config.getLengthPunishingStrategy(), config.getLengthPunishingFactor());
        }

        System.out.println("No solution found in " + maxGeneration + " generations");
        return this.population.getFittest();
    }

    /**
     * Performs crossover between two parent individuals to create a new child.
     * <p>
     * The method of crossover (e.g., one-point, uniform) and the handling of leftover genes
     * from parents of different lengths are determined by the {@link GAConfig}.
     * </p>
     * @param individual1 The first parent.
     * @param individual2 The second parent.
     * @return A new {@link Individual} (child) resulting from the crossover.
     */
    private Individual crossover(Individual individual1, Individual individual2) {
        int len1 = individual1.getGenomeLength();
        int len2 = individual2.getGenomeLength();

        // Guard clause: fallback to the fittest parent for very short genomes
        if (len1 <= 1 || len2 <= 1) {
            return (individual1.getFitness() > individual2.getFitness())
                    ? individual1
                    : individual2;
        }

        CrossoverStrategy crossoverStrategy = config.getCrossoverStrategy();
        CrossoverLeftoverStrategy leftoverStrategy = config.getCrossoverLeftoverStrategy();

        List<Byte> genome1 = individual1.getGenome();
        List<Byte> genome2 = individual2.getGenome();
        int minLength = Math.min(len1, len2);

        List<Byte> newGenome = new ArrayList<>(minLength);

        try {
            switch (crossoverStrategy) {
                case ONE_POINT -> {
                    int cut = random.nextInt(minLength - 1);
                    newGenome.addAll(genome1.subList(0, cut));
                    newGenome.addAll(genome2.subList(cut, len2));
                    return new Individual(newGenome);
                }

                case TWO_POINT -> {
                    int cut1 = random.nextInt(minLength - 1) + 1; // [1, minLength - 1]
                    int cut2 = random.nextInt(minLength - cut1) + cut1; // [cut1, minLength - 1]
                    newGenome.addAll(genome1.subList(0, cut1));
                    newGenome.addAll(genome2.subList(cut1, cut2));
                    newGenome.addAll(genome1.subList(cut2, minLength));
                    return new Individual(newGenome);
                }

                case UNIFORM -> {
                    int pickA = minLength / 2 + (random.nextBoolean() ? minLength % 2 : 0);
                    int pickB = minLength - pickA;

                    for (int i = 0; i < minLength; i++) {
                        boolean chooseA = pickB == 0 || (pickA > 0 && random.nextBoolean());
                        newGenome.add((chooseA ? genome1 : genome2).get(i));
                        if (chooseA) pickA--; else pickB--;
                    }
                }

                case ARITHMETIC -> {
                    for (int i = 0; i < minLength; i++) {
                        newGenome.add((byte) (genome1.get(i) ^ genome2.get(i)));
                    }
                }

                default -> throw new UnsupportedOperationException(
                        "Unknown crossover strategy: " + crossoverStrategy
                );
            }
        }catch (Exception e){
            System.err.println(e.getMessage());
            System.err.println(crossoverStrategy);
            System.err.println(leftoverStrategy);
            System.err.println(individual1);
            System.err.println(individual2);
            System.exit(-1);
        }

        // Common leftover handling (for UNIFORM and ARITHMETIC)
        List<Byte> longer = (genome1.size() > genome2.size()) ? genome1 : genome2;
        List<Byte> leftovers = new ArrayList<>(longer.subList(minLength, longer.size()));

        return this.processLeftovers(leftoverStrategy, newGenome, leftovers, individual1, individual2);
    }

    /**
     * Handles leftover genes from the longer parent's genome after crossover.
     * <p>
     * Based on the configured {@link CrossoverLeftoverStrategy}, this method decides how to
     * append (or not append) the remaining genes to the child's genome.
     * </p>
     *
     * @param strategy The strategy for handling leftover genes.
     * @param newGenome The child's genome being constructed.
     * @param leftovers The list of leftover genes from the longer parent.
     * @param firstIndividual The first parent individual.
     * @param secondIndividual The second parent individual.
     * @return A new {@link Individual} with the final genome after handling leftovers.
     */
    private Individual processLeftovers(CrossoverLeftoverStrategy strategy, List<Byte> newGenome, List<Byte> leftovers, Individual firstIndividual, Individual secondIndividual) {
        if (leftovers.isEmpty()) {
            return new Individual(newGenome);
        }

        List<Byte> genome1 = firstIndividual.getGenome();
        List<Byte> genome2 = secondIndividual.getGenome();
        int minLength = Math.min(genome1.size(), genome2.size());

        switch (strategy) {
            case KEEP_ALL_OR_NOTHING_RANDOMLY -> {
                if (random.nextBoolean()) {
                    newGenome.addAll(leftovers);
                }
            }

            case KEEP_ONE_OR_NOT_RANDOMLY -> {
                for (byte leftover : leftovers) {
                    if (random.nextBoolean()) {
                        newGenome.add(leftover);
                    }
                }
            }

            case KEEP_ONLY_FROM_FITTEST_PARENT -> {
                List<Byte> fittestGenome = (firstIndividual.getFitness() >= secondIndividual.getFitness())
                        ? genome1
                        : genome2;
                newGenome.addAll(fittestGenome.subList(minLength, fittestGenome.size()));
            }

            default -> throw new UnsupportedOperationException(
                    "Unknown crossover leftover strategy: " + strategy
            );
        }

        return new Individual(newGenome);
    }

    /**
     * Applies a mutation to an individual's genome.
     * <p>
     * This method assumes the decision to mutate has already been made. It selects a specific
     * mutation type (ADD, REMOVE, or FLIP) based on the relative weights of their configured rates
     * ({@code bitAddRate}, {@code bitRemoveRate}, {@code bitFlipRate}). The sum of these rates
     * equals the overall {@code mutationRate}.
     * </p>
     * @param individual The individual to mutate.
     * @return The same individual instance, which has been modified in-place.
     */
    private Individual mutate(Individual individual){
        int randomGeneIndex = random.nextInt(individual.getGenomeLength());
        Byte randomGene = (byte) random.nextInt(2);

        // Pick a mutation in a roulette-like fashion
        double pick = random.nextDouble();
        if (pick <= config.getBitAddRate()) {
            // ADD mutation
            individual.addGene(randomGene);
        } else if (pick <= config.getBitAddRate() + config.getBitRemoveRate()) {
            // REMOVE mutation
            if (individual.getGenomeLength() > 1) {
                individual.removeGene(randomGeneIndex);
            }
        } else {
            // FLIP mutation
            individual.setGene(randomGeneIndex, randomGene);
        }
        return individual;
    }

    /**
     * Selects a subset of individuals from a given list based on the configured selection strategy.
     *
     * @param individuals The pool of individuals to select from.
     * @param selectionSize The number of individuals to select.
     * @return A new list containing the selected individuals.
     * @throws UnsupportedOperationException if the configured selection strategy is not recognized.
     * @see SelectionStrategy
     */
    public List<Individual> selection(List<Individual> individuals, int selectionSize) {
        SelectionStrategy selectionStrategy = config.getSelectionStrategy();
        switch (selectionStrategy) {
            case ELITISM:
                // Select the fittest individuals
                Collections.sort(individuals);
                return individuals.subList(0,selectionSize);
            case ROULETTE:
                // Fitness-proportionate selection
                List<Individual> rouletteWinners = new ArrayList<>(selectionSize);

                int totalFitness = 0;
                for (Individual i: individuals) {
                    totalFitness += i.getFitness();
                }

                for (int i = 0; i < selectionSize; i++) {
                    int pick = this.random.nextInt(totalFitness);

                    int rouletteSum = 0;
                    for (Individual individual : individuals) {
                        rouletteSum += individual.getFitness();
                        if (rouletteSum >= pick) {
                            rouletteWinners.add(individual);
                            break;
                        }
                    }
                }
                return rouletteWinners;
            case TOURNAMENT:
                // Select the fittest individuals from a random sample
                int tournamentSize = config.getTournamentSize();
                List<Individual> tournamentWinners = new ArrayList<>(selectionSize);
                for (int i = 0; i < selectionSize; i++) {
                    Collections.shuffle(individuals, random);
                    tournamentWinners.addAll(individuals.subList(0, tournamentSize));
                }
                return tournamentWinners;
        }
        throw new UnsupportedOperationException("selectionStrategy was not ELITISM, ROULETTE or TOURNAMENT");
    }
}