package be.brw.domain;

public class XORNetwork {

    // ===== Existing compute function =====
    private static double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    public static double compute(double x1, double x2,
                                 double w11, double w12, double w21, double w22,
                                 double w3, double w4,
                                 double b1, double b2, double b3) {

        double z11 = x1 * w11 + x2 * w21 + b1;
        double z12 = x1 * w12 + x2 * w22 + b2;

        double a11 = sigmoid(z11);
        double a12 = sigmoid(z12);

        double z2 = a11 * w3 + a12 * w4 + b3;
        return sigmoid(z2);
    }

    // ===== Bitstring decoder + evaluation =====
    public static double evaluate(String bitstring, double x1, double x2) {
        final int bitsPerParam = 8;
        final double min = -5.0;
        final double max = 5.0;

        // We expect exactly 9 parameters * 8 bits = 72 bits
        if (bitstring.length() != bitsPerParam * 9) {
            throw new IllegalArgumentException("Bitstring must be exactly " + (bitsPerParam * 9) + " bits long.");
        }

        double[] params = new double[9];

        for (int i = 0; i < 9; i++) {
            String segment = bitstring.substring(i * bitsPerParam, (i + 1) * bitsPerParam);
            int intVal = Integer.parseInt(segment, 2);
            double value = min + (intVal / 255.0) * (max - min);
            params[i] = value;
        }

        // Decode parameters in order
        double w11 = params[0];
        double w12 = params[1];
        double w21 = params[2];
        double w22 = params[3];
        double b1  = params[4];
        double b2  = params[5];
        double w3  = params[6];
        double w4  = params[7];
        double b3  = params[8];

        // Compute the prediction
        return compute(x1, x2, w11, w12, w21, w22, w3, w4, b1, b2, b3);
    }

    // ===== Example usage =====
    public static void main(String[] args) {
        // Example: random 72-bit chromosome (each param = 8 bits)
        String bitstring = "101101010100101111100010001011011001110011110001010100101011001011001010";
        double output = evaluate(bitstring, 1, 0);
        System.out.println("Prediction for (1,0): " + output);
    }
}
