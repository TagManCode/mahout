package org.apache.mahout.clustering.sequence;

import com.google.common.collect.Lists;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.clustering.iterator.ClusterWritable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Collection;

public class SequenceCanopyReducer extends Reducer<Text, VectorWritable, Text, ClusterWritable> {

    private final Collection<SequenceCanopy> canopies = Lists.newArrayList();
    private SequenceCanopyClusterer canopyClusterer;
    private int clusterFilter;
    private static Logger LOGGER = LoggerFactory.getLogger(SequenceCanopyReducer.class);

    @Override
    protected void reduce(Text arg0, Iterable<VectorWritable> values, Context context) throws IOException, InterruptedException {

        LOGGER.info("Adding Canopies centers to new Canopies");
        for (VectorWritable value : values) {
            Vector point = value.get();
            canopyClusterer.addPointToCanopies(point, canopies);
        }

        LOGGER.info("Get {} new canopies", canopies.size());
        for (SequenceCanopy canopy : canopies) {
            canopy.computeParameters();
            if (canopy.getNumObservations() > clusterFilter) {
                ClusterWritable clusterWritable = new ClusterWritable();
                clusterWritable.setValue(canopy);
                context.write(new Text(canopy.getIdentifier()), clusterWritable);
            }
        }
    }

    @Override
    protected void setup(Context context) throws IOException, InterruptedException {
        canopyClusterer = SequenceCanopyConfigKeys.configureCanopyClusterer(context.getConfiguration());
        canopyClusterer.useT3T4();
        clusterFilter = Integer.parseInt(context.getConfiguration().get(SequenceCanopyConfigKeys.CF_KEY));
    }
}
