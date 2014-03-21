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
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Collection;

public class SequenceCanopyMapper extends Mapper<WritableComparable<?>, VectorWritable, Text, VectorWritable> {

    private final Collection<SequenceCanopy> canopies = Lists.newArrayList();
    private SequenceCanopyClusterer canopyClusterer;
    private int clusterFilter;
    private static Logger LOGGER = LoggerFactory.getLogger(SequenceCanopyMapper.class);

    @Override
    protected void map(WritableComparable<?> key, VectorWritable point, Context context) throws IOException, InterruptedException {
        canopyClusterer.addPointToCanopies(point.get(), canopies);
    }

    @Override
    protected void setup(Context context) throws IOException, InterruptedException {
        super.setup(context);
        canopyClusterer = SequenceCanopyConfigKeys.configureCanopyClusterer(context.getConfiguration());
        clusterFilter = Integer.parseInt(context.getConfiguration().get(SequenceCanopyConfigKeys.CF_KEY));
    }

    @Override
    protected void cleanup(Context context) throws IOException, InterruptedException {
        LOGGER.info("Adding {} new canopies", canopies.size());
        for (SequenceCanopy canopy : canopies) {
            canopy.computeParameters();
            if (canopy.getNumObservations() > clusterFilter) {
                context.write(new Text("centroid"), new VectorWritable(canopy.getCenter()));
            }
        }
    }
}
