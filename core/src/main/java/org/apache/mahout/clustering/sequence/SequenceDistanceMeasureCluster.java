package org.apache.mahout.clustering.sequence;

import org.apache.hadoop.conf.Configuration;
import org.apache.mahout.clustering.Model;
import org.apache.mahout.common.ClassUtils;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

/**
 * Author: antoine.amend@tagman.com
 * Date: 21/03/14
 */
public class SequenceDistanceMeasureCluster extends SequenceAbstractCluster {

    private DistanceMeasure measure;

    public SequenceDistanceMeasureCluster(Vector point, int id, DistanceMeasure measure) {
        super(point, id);
        this.measure = measure;
    }

    public SequenceDistanceMeasureCluster() {
    }

    @Override
    public void configure(Configuration job) {
        if (measure != null) {
            measure.configure(job);
        }
    }

    @Override
    public void readFields(DataInput in) throws IOException {
        String dm = in.readUTF();
        this.measure = ClassUtils.instantiateAs(dm, DistanceMeasure.class);
        super.readFields(in);
    }

    @Override
    public void write(DataOutput out) throws IOException {
        out.writeUTF(measure.getClass().getName());
        super.write(out);
    }

    @Override
    public double pdf(VectorWritable vw) {
        return 1 / (1 + measure.distance(vw.get(), getCenter()));
    }

    @Override
    public Model<VectorWritable> sampleFromPosterior() {
        return new SequenceDistanceMeasureCluster(getCenter(), getId(), measure);
    }

    public DistanceMeasure getMeasure() {
        return measure;
    }

    /**
     * @param measure the measure to set
     */
    public void setMeasure(DistanceMeasure measure) {
        this.measure = measure;
    }

    @Override
    public String getIdentifier() {
        return "DMC:" + getId();
    }

}
