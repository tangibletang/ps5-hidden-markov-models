import java.util.*;
import java.io.*;

/**
 * @Author Alex Tang CS10 Fall 2025
 */
public class HMM {
    // Constant for unseen word observation score (tune this value as needed)
    private static final double UNSEEN_WORD_SCORE = -100.0;

    // Nested maps for counts
    private Map<String, Map<String, Integer>> transitionCounts;
    private Map<String, Map<String, Integer>> observationCounts;

    // Nested maps for log probabilities
    private Map<String, Map<String, Double>> transitionProbs;
    private Map<String, Map<String, Double>> observationProbs;

    // Keep track of all tags
    private Set<String> allTags;

    public HMM() {
        transitionCounts = new HashMap<>();
        observationCounts = new HashMap<>();
        transitionProbs = new HashMap<>();
        observationProbs = new HashMap<>();
        allTags = new HashSet<>();
    }

    /**
     * For Task 1.2: Read a file and return a list of lists of tokens.
     */
    public List<List<String>> readFile(String filename) throws IOException {
        List<List<String>> result = new ArrayList<>();
        BufferedReader reader = new BufferedReader(new FileReader(filename));
        String line;

        while ((line = reader.readLine()) != null) {
            line = line.trim();
            if (!line.isEmpty()) {
                String[] tokens = line.split(" ");
                List<String> tokenList = new ArrayList<>();
                for (String token : tokens) {
                    tokenList.add(token.toLowerCase());
                }
                result.add(tokenList);
            }
        }
        reader.close();
        return result;
    }

    /**
     * For Task 1.2: Read a tags file (no lowercasing for tags).
     */
    public List<List<String>> readTagsFile(String filename) throws IOException {
        List<List<String>> result = new ArrayList<>();
        BufferedReader reader = new BufferedReader(new FileReader(filename));
        String line;

        while ((line = reader.readLine()) != null) {
            line = line.trim();
            if (!line.isEmpty()) {
                String[] tokens = line.split(" ");
                List<String> tagList = new ArrayList<>();
                for (String token : tokens) {
                    tagList.add(token);
                }
                result.add(tagList);
            }
        }
        reader.close();
        return result;
    }

    /**
     * Task 1.1: Train on lists of words and corresponding lists of tags.
     */
    public void train(List<List<String>> sentences, List<List<String>> tagSequences) {
        // Count observations and transitions
        for (int i = 0; i < sentences.size(); i++) {
            List<String> words = sentences.get(i);
            List<String> tags = tagSequences.get(i);

            String prevTag = "<START>";

            for (int j = 0; j < words.size(); j++) {
                String word = words.get(j);
                String tag = tags.get(j);

                // Track all tags
                allTags.add(tag);

                // Count transition
                if (!transitionCounts.containsKey(prevTag)) {
                    transitionCounts.put(prevTag, new HashMap<>());
                }
                Map<String, Integer> transitions = transitionCounts.get(prevTag);
                if (transitions.containsKey(tag)) {
                    transitions.put(tag, transitions.get(tag) + 1);
                } else {
                    transitions.put(tag, 1);
                }

                // Count observation
                if (!observationCounts.containsKey(tag)) {
                    observationCounts.put(tag, new HashMap<>());
                }
                Map<String, Integer> observations = observationCounts.get(tag);
                if (observations.containsKey(word)) {
                    observations.put(word, observations.get(word) + 1);
                } else {
                    observations.put(word, 1);
                }

                prevTag = tag;
            }
        }

        // Convert counts to log probabilities
        countsToLogProbs();
    }

    /**
     * Convert all counts to log probabilities.
     */
    private void countsToLogProbs() {
        // Transition probabilities
        for (String prevTag : transitionCounts.keySet()) {
            Map<String, Integer> transitions = transitionCounts.get(prevTag);
            Map<String, Double> logProbs = new HashMap<>();

            int total = 0;
            for (int count : transitions.values()) {
                total += count;
            }

            for (String currTag : transitions.keySet()) {
                int count = transitions.get(currTag);
                logProbs.put(currTag, Math.log((double) count / total));
            }

            transitionProbs.put(prevTag, logProbs);
        }

        // Observation probabilities
        for (String tag : observationCounts.keySet()) {
            Map<String, Integer> observations = observationCounts.get(tag);
            Map<String, Double> logProbs = new HashMap<>();

            int total = 0;
            for (int count : observations.values()) {
                total += count;
            }

            for (String word : observations.keySet()) {
                int count = observations.get(word);
                logProbs.put(word, Math.log((double) count / total));
            }

            observationProbs.put(tag, logProbs);
        }
    }

    public double getTransitionProb(String prevTag, String currTag) {
        if (transitionProbs.containsKey(prevTag) &&
                transitionProbs.get(prevTag).containsKey(currTag)) {
            return transitionProbs.get(prevTag).get(currTag);
        }
        return Double.NEGATIVE_INFINITY;
    }

    public double getObservationProb(String tag, String word) {
        if (observationProbs.containsKey(tag) &&
                observationProbs.get(tag).containsKey(word)) {
            return observationProbs.get(tag).get(word);
        }
        return Double.NEGATIVE_INFINITY;
    }

    /**
     * Task 2.1: Viterbi algorithm to find the best tag sequence for a sentence.
     *
     * @param sentence list of words (should be lowercased)
     * @return list of tags corresponding to each word
     */
    public List<String> viterbi(List<String> sentence) {
        // currStates = { start }
        Set<String> currStates = new HashSet<>();
        currStates.add("<START>");

        // currScores = map { start=0 }
        Map<String, Double> currScores = new HashMap<>();
        currScores.put("<START>", 0.0);

        // backtrace = empty list
        List<Map<String, String>> backtrace = new ArrayList<>();

        // for i from 0 to # observations - 1
        for (int i = 0; i < sentence.size(); i++) {
            String observation = sentence.get(i);

            // nextStates = {}
            Set<String> nextStates = new HashSet<>();

            // nextScores = empty map
            Map<String, Double> nextScores = new HashMap<>();

            // backtrace[i] = empty map
            Map<String, String> backtraceStep = new HashMap<>();

            // for each currState in currStates
            for (String currState : currStates) {

                // for each transition currState -> nextState
                if (!transitionProbs.containsKey(currState)) {
                    continue;
                }

                Map<String, Double> transitions = transitionProbs.get(currState);

                for (String nextState : transitions.keySet()) {
                    // add nextState to nextStates
                    nextStates.add(nextState);

                    // nextScore = currScores[currState] +
                    //             transitionScore(currState -> nextState) +
                    //             observationScore(observations[i] in nextState)
                    double transitionScore = getTransitionProb(currState, nextState);
                    double observationScore = getObservationProb(nextState, observation);

                    // Use constant for unseen words
                    if (observationScore == Double.NEGATIVE_INFINITY) {
                        observationScore = UNSEEN_WORD_SCORE;
                    }

                    double nextScore = currScores.get(currState) + transitionScore + observationScore;

                    // if nextState isn't in nextScores or nextScore > nextScores[nextState]
                    if (!nextScores.containsKey(nextState) || nextScore > nextScores.get(nextState)) {
                        // set nextScores[nextState] to nextScore
                        nextScores.put(nextState, nextScore);

                        // set backtrace[i][nextState] to currState
                        backtraceStep.put(nextState, currState);
                    }
                }
            }

            // Add backtrace for this position
            backtrace.add(backtraceStep);

            // currStates = nextStates
            currStates = nextStates;

            // currScores = nextScores
            currScores = nextScores;
        }

        // Find the state with highest score in currStates
        String bestFinalState = null;
        double bestScore = Double.NEGATIVE_INFINITY;

        for (String state : currStates) {
            double score = currScores.get(state);
            if (score > bestScore) {
                bestScore = score;
                bestFinalState = state;
            }
        }

        // Backtrace from bestFinalState back to start
        List<String> tags = new ArrayList<>();
        String currentState = bestFinalState;

        // Go backwards through backtrace
        for (int i = backtrace.size() - 1; i >= 0; i--) {
            tags.add(0, currentState); // Add to front
            currentState = backtrace.get(i).get(currentState);
        }

        return tags;
    }

    /**
     * Test the HMM on a file of test sentences.
     */
    public void testOnFile(String testFile) throws IOException {
        List<List<String>> testSentences = readFile(testFile);

        System.out.println("\nTesting on " + testSentences.size() + " sentences:");
        System.out.println("=" .repeat(60));

        for (int i = 0; i < Math.min(5, testSentences.size()); i++) {
            List<String> sentence = testSentences.get(i);
            List<String> tags = viterbi(sentence);

            System.out.println("\nSentence " + (i + 1) + ":");
            for (int j = 0; j < sentence.size(); j++) {
                System.out.print(sentence.get(j) + "/" + tags.get(j) + " ");
            }
            System.out.println();
        }
    }

    public static void main(String[] args) {
        try {
            HMM hmm = new HMM();

            // Task 1: Train
            System.out.println("Training...");
            List<List<String>> trainSentences = hmm.readFile("/Users/alex/Desktop/CS10/CS10/Inputs/brown-train-sentences.txt");
            List<List<String>> trainTags = hmm.readTagsFile("/Users/alex/Desktop/CS10/CS10/Inputs/brown-train-tags.txt");
            hmm.train(trainSentences, trainTags);
            System.out.println("Training complete!");

            // Task 2.2: Test
            System.out.println("\nTesting...");
            List<List<String>> testSentences = hmm.readFile("/Users/alex/Desktop/CS10/CS10/Inputs/brown-test-sentences.txt");
            List<List<String>> testTags = hmm.readTagsFile("/Users/alex/Desktop/CS10/CS10/Inputs/brown-test-tags.txt");

            int correct = 0;
            int incorrect = 0;

            for (int i = 0; i < testSentences.size(); i++) {
                List<String> sentence = testSentences.get(i);
                List<String> correctTags = testTags.get(i);
                List<String> predictedTags = hmm.viterbi(sentence);

                for (int j = 0; j < correctTags.size(); j++) {
                    if (predictedTags.get(j).equals(correctTags.get(j))) {
                        correct++;
                    } else {
                        incorrect++;
                    }
                }
            }

            System.out.println("Correct: " + correct);
            System.out.println("Incorrect: " + incorrect);
            System.out.println("Accuracy: " + (100.0 * correct / (correct + incorrect)) + "%");

        } catch (IOException e) {
            System.err.println("Error: " + e.getMessage());
        }
    }
}