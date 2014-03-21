/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mahout.clustering.sequence;

import com.google.common.collect.Lists;
import org.apache.mahout.clustering.AbstractCluster;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.math.Vector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collection;
import java.util.Iterator;
import java.util.List;

public class SequenceCanopyClusterer {

    private static final Logger log = LoggerFactory.getLogger(SequenceCanopyClusterer.class);

    private int nextCanopyId;
    private double t1;
    private double t2;
    private double t3;
    private double t4;
    private DistanceMeasure measure;

    public SequenceCanopyClusterer(DistanceMeasure measure, double t1, double t2) {
        this.t1 = t1;
        this.t2 = t2;
        this.t3 = t1;
        this.t4 = t2;
        this.measure = measure;
    }

    public double getT1() {
        return t1;
    }

    public double getT2() {
        return t2;
    }

    public double getT3() {
        return t3;
    }

    public double getT4() {
        return t4;
    }

    /**
     * Used by CanopyReducer to set t1=t3 and t2=t4 configuration values
     */
    public void useT3T4() {
        t1 = t3;
        t2 = t4;
    }

    /**
     * This is the same algorithm as the reference but inverted to iterate over
     * existing canopies instead of the points. Because of this it does not need
     * to actually store the points, instead storing a total points vector and
     * the number of points. From this a centroid can be computed.
     * <p/>
     * This method is used by the SequenceCanopyMapper, CanopyReducer and CanopyDriver.
     *
     * @param point    the point to be added
     * @param canopies the List<Canopy> to be appended
     */
    public void addPointToCanopies(Vector point, Collection<SequenceCanopy> canopies) {
        boolean pointStronglyBound = false;
        for (SequenceCanopy canopy : canopies) {
            double dist = measure.distance(canopy.getCenter(), point);
            if (dist < t1) {
                //log.info("Added point: {} to canopy: {}", SequenceAbstractCluster.formatVector(point, null), canopy.getIdentifier());
                canopy.observe(point);
            }
            pointStronglyBound = pointStronglyBound || dist < t2;
        }
        if (!pointStronglyBound) {
            //log.info("Created new Canopy:{} at center:{}", nextCanopyId, SequenceAbstractCluster.formatVector(point, null));
            canopies.add(new SequenceCanopy(point, nextCanopyId++, measure));
        }
    }

    /**
     * Iterate through the points, adding new canopies. Return the canopies.
     *
     * @param points  a list<Vector> defining the points to be clustered
     * @param measure a DistanceMeasure to use
     * @param t1      the T1 distance threshold
     * @param t2      the T2 distance threshold
     * @return the List<Canopy> created
     */
    public static List<SequenceCanopy> createCanopies(List<Vector> points, DistanceMeasure measure, double t1, double t2) {
        List<SequenceCanopy> canopies = Lists.newArrayList();
        /**
         * Reference Implementation: Given a distance metric, one can create
         * canopies as follows: Start with a list of the data points in any
         * order, and with two distance thresholds, T1 and T2, where T1 > T2.
         * (These thresholds can be set by the user, or selected by
         * cross-validation.) Pick a point on the list and measure its distance
         * to all other points. Put all points that are within distance
         * threshold T1 into a canopy. Remove from the list all points that are
         * within distance threshold T2. Repeat until the list is empty.
         */
        int nextCanopyId = 0;
        while (!points.isEmpty()) {
            Iterator<Vector> ptIter = points.iterator();
            Vector p1 = ptIter.next();
            ptIter.remove();
            SequenceCanopy canopy = new SequenceCanopy(p1, nextCanopyId++, measure);
            canopies.add(canopy);
            while (ptIter.hasNext()) {
                Vector p2 = ptIter.next();
                double dist = measure.distance(p1, p2);
                // Put all points that are within distance threshold T1 into the
                // canopy
                if (dist < t1) {
                    canopy.observe(p2);
                }
                // Remove from the list all points that are within distance
                // threshold T2
                if (dist < t2) {
                    ptIter.remove();
                }
            }
            for (SequenceCanopy c : canopies) {
                c.computeParameters();
            }
        }
        return canopies;
    }

    public void setT3(double t3) {
        this.t3 = t3;
    }

    public void setT4(double t4) {
        this.t4 = t4;
    }
}
