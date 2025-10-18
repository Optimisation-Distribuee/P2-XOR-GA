package be.brw;

import be.brw.config.ConfigLoader;
import be.brw.config.GAConfig;
import be.brw.domain.GeneticAlgorithm;
import be.brw.domain.Individual;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.*;
import java.time.LocalDateTime;
import java.util.*;

public class Check {
    public static void main(String[] args) {
        Path configDir = Path.of("src/main/resources/configs");
        Path logFile = Path.of("results.log");

        class Result {
            String genome;
            int generations;
        }

        List<Result> results = new ArrayList<>();

        try (BufferedWriter writer = new BufferedWriter(new FileWriter(logFile.toFile(), false))) {
            writer.write("===== Genetic Algorithm Benchmark =====\n");
            writer.write("Started at: " + LocalDateTime.now() + "\n\n");

            // Find all YAML configs
            List<Path> configs;
            try (var stream = Files.list(configDir)) {
                configs = stream
                        .filter(p -> p.toString().endsWith(".yaml"))
                        .sorted()
                        .toList();
            }

            if (configs.isEmpty()) {
                writer.write("No config files found in " + configDir.toAbsolutePath() + "\n");
                return;
            }

            // Run each config
            for (Path configPath : configs) {
                try {
                    GAConfig config = ConfigLoader.fromYaml(configPath);
                    GeneticAlgorithm ga = new GeneticAlgorithm(config);

                    Individual result = ga.runAlgorithm();
                    int generationCount = ga.getGenerationCount();

                    String line = String.format(
                            "[%s] Fitness=%d  Generations=%d%n",
                            configPath.getFileName(),
                            result.getFitness(),
                            generationCount
                    );
                    writer.write(line);
                    // System.out.print(line);

                    // Keep successful XOR solvers (fitness == 4)
                    if (result.getFitness() == 4) {
                        StringBuilder bitstring = new StringBuilder();
                        for (Byte b : result.getGenome()) {
                            bitstring.append(b == 0 ? '0' : '1');
                        }

                        Result r = new Result();
                        r.genome = bitstring.toString();
                        r.generations = generationCount;
                        results.add(r);
                    }

                } catch (Exception e) {
                    writer.write(String.format("[%s] FAILED: %s%n", configPath.getFileName(), e.getMessage()));
                    System.err.printf("[%s] FAILED: %s%n", configPath.getFileName(), e.getMessage());
                }
            }

            // Remove duplicate genomes, keeping the one with lowest generation count
            Map<String, Result> unique = new HashMap<>();
            for (Result r : results) {
                unique.merge(r.genome, r, (oldR, newR) ->
                        (newR.generations < oldR.generations) ? newR : oldR
                );
            }

            // Sort unique results by generation count
            List<Result> uniqueResults = new ArrayList<>(unique.values());
            uniqueResults.sort(Comparator.comparingInt(r -> r.generations));

            // Print summary
            writer.write("\n===== Solutions =====\n");
            for (Result r : uniqueResults) {
                String summary = String.format("'%s' GEN=%d%n", r.genome, r.generations);
                writer.write(summary);
                System.out.print(summary);
            }

            writer.write("\n===== End of Benchmark =====\n");
            System.out.println("\nResults written to " + logFile.toAbsolutePath());

        } catch (IOException e) {
            System.err.println("Error during benchmark: " + e.getMessage());
        }
    }
}
