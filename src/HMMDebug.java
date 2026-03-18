import java.util.*;
import java.io.*;

public class HMMDebug {
    
    public static void main(String[] args) {
        try {
            System.out.println("=".repeat(70));
            System.out.println("HMM DEBUGGING TEST");
            System.out.println("=".repeat(70));
            
            HMM hmm = new HMM();
            
            // ===== STEP 1: TRAINING =====
            System.out.println("\n[STEP 1] TRAINING");
            System.out.println("-".repeat(70));

            List<List<String>> trainSentences = hmm.readFile("/Users/alex/Desktop/CS10/CS10/Inputs/brown-train-sentences.txt");
            List<List<String>> trainTags = hmm.readTagsFile("/Users/alex/Desktop/CS10/CS10/Inputs/brown-train-tags.txt");
            
            System.out.println("✓ Read " + trainSentences.size() + " training sentences");
            System.out.println("✓ Read " + trainTags.size() + " training tag sequences");
            
            // Check first training sentence
            if (trainSentences.size() > 0) {
                System.out.println("\nFirst training sentence:");
                System.out.println("  Words: " + trainSentences.get(0));
                System.out.println("  Tags:  " + trainTags.get(0));
            }
            
            System.out.println("\nTraining model...");
            hmm.train(trainSentences, trainTags);
            System.out.println("✓ Training complete!");
            
            // ===== STEP 2: VERIFY TRAINING =====
            System.out.println("\n[STEP 2] VERIFY TRAINING WORKED");
            System.out.println("-".repeat(70));
            
            System.out.println("\nSample transition probabilities:");
            System.out.println("  P(DET | <START>): " + hmm.getTransitionProb("<START>", "DET"));
            System.out.println("  P(N | DET):       " + hmm.getTransitionProb("DET", "N"));
            System.out.println("  P(V | N):         " + hmm.getTransitionProb("N", "V"));
            System.out.println("  P(ADJ | DET):     " + hmm.getTransitionProb("DET", "ADJ"));
            
            System.out.println("\nSample observation probabilities:");
            System.out.println("  P('the' | DET):   " + hmm.getObservationProb("DET", "the"));
            System.out.println("  P('a' | DET):     " + hmm.getObservationProb("DET", "a"));
            System.out.println("  P('dog' | N):     " + hmm.getObservationProb("N", "dog"));
            System.out.println("  P('run' | V):     " + hmm.getObservationProb("V", "run"));
            
            // Check for -Infinity (bad!)
            boolean hasValidProbs = hmm.getTransitionProb("DET", "N") != Double.NEGATIVE_INFINITY;
            if (!hasValidProbs) {
                System.out.println("\n⚠️  WARNING: All probabilities are -Infinity!");
                System.out.println("   Training may have failed!");
                return;
            }
            
            // ===== STEP 3: LOAD TEST DATA =====
            System.out.println("\n[STEP 3] LOADING TEST DATA");
            System.out.println("-".repeat(70));
            
            List<List<String>> testSentences = hmm.readFile("/Users/alex/Desktop/CS10/CS10/Inputs/brown-test-sentences.txt");
            List<List<String>> testTags = hmm.readTagsFile("/Users/alex/Desktop/CS10/CS10/Inputs/brown-test-tags.txt");
            
            System.out.println("✓ Read " + testSentences.size() + " test sentences");
            System.out.println("✓ Read " + testTags.size() + " test tag sequences");
            
            // ===== STEP 4: TEST VITERBI ON FIRST FEW SENTENCES =====
            System.out.println("\n[STEP 4] TESTING VITERBI ON SAMPLE SENTENCES");
            System.out.println("-".repeat(70));
            
            int samplesToShow = Math.min(5, testSentences.size());
            for (int i = 0; i < samplesToShow; i++) {
                List<String> sentence = testSentences.get(i);
                List<String> correctTags = testTags.get(i);
                List<String> predictedTags = hmm.viterbi(sentence);
                
                System.out.println("\nTest Sentence #" + (i + 1) + ":");
                System.out.println("  Words:     " + sentence);
                System.out.println("  Correct:   " + correctTags);
                System.out.println("  Predicted: " + predictedTags);
                
                // Check lengths
                if (predictedTags.size() != correctTags.size()) {
                    System.out.println("  ⚠️  LENGTH MISMATCH! Predicted: " + predictedTags.size() + 
                                     ", Correct: " + correctTags.size());
                }
                
                // Count matches in this sentence
                int matches = 0;
                for (int j = 0; j < Math.min(predictedTags.size(), correctTags.size()); j++) {
                    if (predictedTags.get(j).equals(correctTags.get(j))) {
                        matches++;
                    }
                }
                System.out.println("  Accuracy:  " + matches + "/" + correctTags.size() + 
                                 " (" + String.format("%.1f", 100.0 * matches / correctTags.size()) + "%)");
            }
            
            // ===== STEP 5: FULL ACCURACY TEST =====
            System.out.println("\n[STEP 5] FULL ACCURACY TEST");
            System.out.println("-".repeat(70));
            
            int correct = 0;
            int incorrect = 0;
            int sizeMismatches = 0;
            
            for (int i = 0; i < testSentences.size(); i++) {
                List<String> sentence = testSentences.get(i);
                List<String> correctTags = testTags.get(i);
                List<String> predictedTags = hmm.viterbi(sentence);
                
                // Check for size mismatch
                if (predictedTags.size() != correctTags.size()) {
                    sizeMismatches++;
                    System.out.println("⚠️  Size mismatch at sentence " + i + 
                                     ": predicted=" + predictedTags.size() + 
                                     ", correct=" + correctTags.size());
                    continue;
                }
                
                // Compare tags
                for (int j = 0; j < correctTags.size(); j++) {
                    if (predictedTags.get(j).equals(correctTags.get(j))) {
                        correct++;
                    } else {
                        incorrect++;
                    }
                }
            }
            
            // ===== STEP 6: RESULTS =====
            System.out.println("\n[STEP 6] FINAL RESULTS");
            System.out.println("=".repeat(70));
            
            int total = correct + incorrect;
            double accuracy = 100.0 * correct / total;
            
            System.out.println("\nCorrect tags:      " + correct);
            System.out.println("Incorrect tags:    " + incorrect);
            System.out.println("Total tags:        " + total);
            System.out.println("Size mismatches:   " + sizeMismatches);
            System.out.println("\nAccuracy:          " + String.format("%.2f", accuracy) + "%");
            System.out.println("Error rate:        " + String.format("%.2f", 100 - accuracy) + "%");
            
            // ===== STEP 7: DIAGNOSIS =====
            System.out.println("\n[STEP 7] DIAGNOSIS");
            System.out.println("=".repeat(70));
            
            if (accuracy < 50) {
                System.out.println("❌ VERY LOW ACCURACY (<50%)");
                System.out.println("   Possible issues:");
                System.out.println("   - Viterbi algorithm bug");
                System.out.println("   - Training data not loaded correctly");
                System.out.println("   - Tag case mismatch (uppercase vs lowercase)");
            } else if (accuracy < 80) {
                System.out.println("⚠️  LOW ACCURACY (50-80%)");
                System.out.println("   Possible issues:");
                System.out.println("   - UNSEEN_WORD_SCORE may need tuning");
                System.out.println("   - Some tags rarely seen in training");
            } else if (accuracy < 95) {
                System.out.println("✓ GOOD ACCURACY (80-95%)");
                System.out.println("   This is expected! Try tuning UNSEEN_WORD_SCORE");
            } else {
                System.out.println("✓✓ EXCELLENT ACCURACY (>95%)");
                System.out.println("   Your model is working well!");
            }
            
            System.out.println("\nExpected results (from assignment):");
            System.out.println("   Brown corpus: ~35109 correct, ~1285 incorrect (96.5% accuracy)");
            System.out.println("   With UNSEEN_WORD_SCORE = -100");
            
        } catch (IOException e) {
            System.err.println("\n❌ FILE ERROR: " + e.getMessage());
            System.err.println("\nMake sure these files exist in the same directory:");
            System.err.println("  - brown-train-sentences.txt");
            System.err.println("  - brown-train-tags.txt");
            System.err.println("  - brown-test-sentences.txt");
            System.err.println("  - brown-test-tags.txt");
            e.printStackTrace();
        } catch (Exception e) {
            System.err.println("\n❌ UNEXPECTED ERROR: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
