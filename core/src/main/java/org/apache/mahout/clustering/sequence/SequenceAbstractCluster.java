package org.apache.mahout.clustering.sequence;

import org.apache.hadoop.conf.Configuration;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.Model;
import org.apache.mahout.common.parameters.Parameter;
import org.apache.mahout.math.*;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Collection;
import java.util.Collections;
import java.util.Locale;

public abstract class SequenceAbstractCluster implements Cluster {

    // cluster persistent state
    private int id;
    private long numObservations;
    private long totalObservations;
    private Vector center;
    private Vector radius;
    private double s0;
    private Vector s1;
    private Vector s2;

    protected SequenceAbstractCluster() {
    }

    protected SequenceAbstractCluster(Vector point, int id2) {
        setNumObservations(0);
        setTotalObservations(0);
        setCenter(point.clone());
        setRadius(center.like());
        setS0(0);
        setS1(center.like());
        setS2(center.like());
        this.id = id2;
    }

    protected SequenceAbstractCluster(Vector center2, Vector radius2, int id2) {
        setNumObservations(0);
        setTotalObservations(0);
        setCenter(new RandomAccessSparseVector(center2));
        setRadius(new RandomAccessSparseVector(radius2));
        setS0(0);
        setS1(center.like());
        setS2(center.like());
        this.id = id2;
    }

    @Override
    public void write(DataOutput out) throws IOException {
        out.writeInt(id);
        out.writeLong(getNumObservations());
        out.writeLong(getTotalObservations());
        VectorWritable.writeVector(out, getCenter());
        VectorWritable.writeVector(out, getRadius());
        out.writeDouble(s0);
        VectorWritable.writeVector(out, s1);
        VectorWritable.writeVector(out, s2);
    }

    @Override
    public void readFields(DataInput in) throws IOException {
        this.id = in.readInt();
        this.setNumObservations(in.readLong());
        this.setTotalObservations(in.readLong());
        this.setCenter(VectorWritable.readVector(in));
        this.setRadius(VectorWritable.readVector(in));
        this.setS0(in.readDouble());
        this.setS1(VectorWritable.readVector(in));
        this.setS2(VectorWritable.readVector(in));
    }

    @Override
    public void configure(Configuration job) {
        // nothing to do
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
    public int getId() {
        return id;
    }

    /**
     * @param id the id to set
     */
    protected void setId(int id) {
        this.id = id;
    }

    @Override
    public long getNumObservations() {
        return numObservations;
    }

    /**
     * @param l the numPoints to set
     */
    protected void setNumObservations(long l) {
        this.numObservations = l;
    }

    @Override
    public long getTotalObservations() {
        return totalObservations;
    }

    protected void setTotalObservations(long totalPoints) {
        this.totalObservations = totalPoints;
    }

    @Override
    public Vector getCenter() {
        return center;
    }

    /**
     * @param center the center to set
     */
    protected void setCenter(Vector center) {
        this.center = center;
    }

    @Override
    public Vector getRadius() {
        return radius;
    }

    /**
     * @param radius the radius to set
     */
    protected void setRadius(Vector radius) {
        this.radius = radius;
    }

    /**
     * @return the s0
     */
    protected double getS0() {
        return s0;
    }

    protected void setS0(double s0) {
        this.s0 = s0;
    }

    /**
     * @return the s1
     */
    protected Vector getS1() {
        return s1;
    }

    protected void setS1(Vector s1) {
        this.s1 = s1;
    }

    /**
     * @return the s2
     */
    protected Vector getS2() {
        return s2;
    }

    protected void setS2(Vector s2) {
        this.s2 = s2;
    }

    @Override
    public void observe(Model<VectorWritable> x) {
        SequenceAbstractCluster cl = (SequenceAbstractCluster) x;
        setS0(cl.getS0());
        setS1(cl.getS1());
        setS2(cl.getS2());
    }

    @Override
    public void observe(VectorWritable x) {
        observe(x.get());
    }

    @Override
    public void observe(VectorWritable x, double weight) {
        observe(x);
    }

    public void observe(Vector x) {
        setS0(getS0() + 1);
        if (getS1() == null) {
            setS1(x.clone());
        }

        if (getS2() == null) {
            setS2(x.clone());
        }
    }


    @Override
    public void computeParameters() {
        if (getS0() == 0) {
            return;
        }
        setNumObservations((long) getS0());
        setTotalObservations(getTotalObservations() + getNumObservations());
        setCenter(getS1());
        setRadius(getS2());
        setS0(0);
        setS1(center.like());
        setS2(center.like());
    }

    @Override
    public String asFormatString(String[] bindings) {
        StringBuilder buf = new StringBuilder(50);
        buf.append(getIdentifier()).append("{n=").append(getNumObservations());
        if (getCenter() != null) {
            buf.append(" c=").append(formatVector(getCenter(), bindings));
        }
        buf.append('}');
        return buf.toString();
    }

    public abstract String getIdentifier();

    /**
     * Compute the centroid by averaging the pointTotals
     *
     * @return the new centroid
     */
    public Vector computeCentroid() {
        return getS0() == 0 ? getCenter() : getS1();
    }

    /**
     * Return a human-readable formatted string representation of the vector, not
     * intended to be complete nor usable as an input/output representation
     */
    public static String formatVector(Vector v, String[] bindings) {
        StringBuilder buffer = new StringBuilder();
        if (v instanceof NamedVector) {
            buffer.append(((NamedVector) v).getName()).append(" = ");
        }

        boolean hasBindings = bindings != null;
        boolean isSparse = !v.isDense() && v.getNumNondefaultElements() != v.size();

        // we assume sequential access in the output
        Vector provider = v.isSequentialAccess() ? v : new SequentialAccessSparseVector(v);

        buffer.append('[');
        for (Vector.Element elem : provider.nonZeroes()) {

            if (hasBindings && bindings.length >= elem.index() + 1 && bindings[elem.index()] != null) {
                buffer.append(bindings[elem.index()]).append(':');
            } else if (hasBindings || isSparse) {
                buffer.append(elem.index()).append(':');
            }

            buffer.append(String.format(Locale.ENGLISH, "%.3f", elem.get())).append(", ");
        }

        if (buffer.length() > 1) {
            buffer.setLength(buffer.length() - 2);
        }
        buffer.append(']');
        return buffer.toString();
    }

    @Override
    public boolean isConverged() {
        // Convergence has no meaning yet, perhaps in subclasses
        return false;
    }
}