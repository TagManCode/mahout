package org.apache.mahout.common.distance;

import org.apache.hadoop.conf.Configuration;
import org.apache.mahout.common.parameters.Parameter;
import org.apache.mahout.math.Vector;

import java.util.*;

/**
 * Author: antoine.amend@tagman.com
 * Date: 10/03/14
 */
public class LevensteinDistanceMeasure implements DistanceMeasure {

    public static double getLevenshteinDistance(List<Double> s, List<Double> t, int threshold) {

        if (s == null || t == null) {
            throw new IllegalArgumentException("Lists must not be null");
        }

        if (threshold < 0) {
            throw new IllegalArgumentException("Threshold must not be negative");
        }

        /*
        This implementation only computes the distance if it's less than or equal to the
        threshold value, returning -1 if it's greater.
        */

        int n = s.size(); // length of s
        int m = t.size(); // length of t

        // if one string is empty, the edit distance is necessarily the length of the other
        if (n == 0) {
            return m <= threshold ? m : 1;
        } else if (m == 0) {
            return n <= threshold ? n : 1;
        }

        if (n > m) {
            // swap the two strings to consume less memory
            final List<Double> tmp = s;
            s = t;
            t = tmp;
            n = m;
            m = t.size();
        }

        int p[] = new int[n + 1]; // 'previous' cost array, horizontally
        int d[] = new int[n + 1]; // cost array, horizontally
        int _d[]; // placeholder to assist in swapping p and d

        // fill in starting table values
        final int boundary = Math.min(n, threshold) + 1;
        for (int i = 0; i < boundary; i++) {
            p[i] = i;
        }
        // these fills ensure that the value above the rightmost entry of our
        // stripe will be ignored in following loop iterations
        Arrays.fill(p, boundary, p.length, Integer.MAX_VALUE);
        Arrays.fill(d, Integer.MAX_VALUE);

        // iterates through t
        for (int j = 1; j <= m; j++) {
            final double t_j = t.get(j - 1); // jth character of t
            d[0] = j;

            // compute stripe indices, constrain to array size
            final int min = Math.max(1, j - threshold);
            final int max = (j > Integer.MAX_VALUE - threshold) ? n : Math.min(n, j + threshold);

            // the stripe may lead off of the table if s and t are of different sizes
            if (min > max) {
                return 1.0;
            }

            // ignore entry left of leftmost
            if (min > 1) {
                d[min - 1] = Integer.MAX_VALUE;
            }

            // iterates through [min, max] in s
            for (int i = min; i <= max; i++) {
                if (s.get(i - 1) == t_j) {
                    // diagonally left and up
                    d[i] = p[i - 1];
                } else {
                    // 1 + minimum of cell to the left, to the top, diagonally left and up
                    d[i] = 1 + Math.min(Math.min(d[i - 1], p[i]), p[i - 1]);
                }
            }

            // copy current distance counts to 'previous row' distance counts
            _d = p;
            p = d;
            d = _d;
        }

        // if p[n] is greater than the threshold, there's no guarantee on it being the correct
        // distance
        if (p[n] <= threshold) {
            double lev = (double) p[n] / (Math.max(s.size(), t.size()));
            return lev;
        } else {
            return 1.0;
        }
    }


    @Override
    public double distance(Vector v1, Vector v2) {

        List<Double> l1 = new ArrayList<Double>(10);
        List<Double> l2 = new ArrayList<Double>(10);

        // We know we don't have a lot of campaigns in user journeys (a dozen at really most)
        // We can afford playing with list (much more convenient)

        for (Vector.Element el1 : v1.nonZeroes()) {
            l1.add(el1.get());
        }

        for (Vector.Element el2 : v2.nonZeroes()) {
            l2.add(el2.get());
        }

        // We consider any sequence more than X% different as totally different (i.e. 1.0)
        // That's should help us getting better performance
        int max = Math.max(l1.size(), l2.size());
        int threshold = (int) Math.ceil(max * 0.2);
        return getLevenshteinDistance(l1, l2, threshold);
    }

    @Override
    public double distance(double centroidLengthSquare, Vector centroid, Vector v) {
        return distance(centroid, v);
    }

    @Override
    public Collection<Parameter<?>> getParameters() {
        return Collections.emptyList();
    }

    @Override
    public void createParameters(String prefix, Configuration jobConf) {
        // nothing to do
    }

    @Override
    public void configure(Configuration config) {
        // nothing to do
    }
}
