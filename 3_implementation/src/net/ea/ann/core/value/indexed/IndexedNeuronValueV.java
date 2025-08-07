/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.core.value.indexed;

import net.ea.ann.core.Util;
import net.ea.ann.core.function.Function;
import net.ea.ann.core.function.FunctionInvertible;
import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.core.value.NeuronValueV;
import net.ea.ann.core.value.WeightValue;

/**
 * This class represents a vector indexed neuron value.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class IndexedNeuronValueV implements IndexedNeuronValue {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Array of values.
	 */
	private NeuronValueV[] values = null;

	
	/**
	 * Internal index.
	 */
	private int index = 0;
	
	
	/**
	 * Constructor with size, dimension, and initial value.
	 * @param size specified size.
	 * @param dim specified dimension.
	 * @param initialValue initial value.
	 */
	public IndexedNeuronValueV(int size, int dim, double initialValue) {
		size = size < 0 ? 0 : size;
		dim = dim < 0 ? 0 : size;
		values = new NeuronValueV[size];
		for (int i = 0; i < values.length; i++) values[i] = new NeuronValueV(dim, initialValue);
	}

	
	/**
	 * Constructor with size and dimension.
	 * @param size specified size.
	 * @param dim specified dimension.
	 */
	public IndexedNeuronValueV(int size, int dim) {
		this(size, dim, 0);
	}

	
	@Override
	public NeuronValue zero() {
		return re(v().zero());
	}

	
	@Override
	public NeuronValue unit() {
		return re(v().unit());
	}

	
	@Override
	public int length() {
		return v().length();
	}

	
	@Override
	public int dim() {
		return length();
	}
	
	
	@Override
	public NeuronValue resize(int newDim) {
		return re(v().resize(newDim));
	}

	
	@Override
	public NeuronValue duplicate() {
		return re(v().duplicate());
	}

	
	@Override
	public boolean equals(NeuronValue value) {
		if (value == null || !(value instanceof IndexedNeuronValue)) return false;
		return v().equals(((IndexedNeuronValue)value).v());
	}

	
	@Override
	public WeightValue newWeightValue() {
		IndexedWeightValueV weight = new IndexedWeightValueV(values.length, values[0].length(), 0);
		weight.setIndex(index);
		return weight;
	}

	
	@Override
	public WeightValue toWeightValue() {
		int size = this.size();
		IndexedWeightValueV neuronValue = new IndexedWeightValueV(size, 1);
		for (int i = 0; i < size; i++) neuronValue.set(i, this.get(i).toWeightValue());
		neuronValue.setIndex(this.getIndex());
		return neuronValue;
	}

	
	@Override
	public NeuronValue negative() {
		return re(v().negative());
	}

	
	@Override
	public boolean canInvert() {
		return v().canInvert();
	}

	
	@Override
	public NeuronValue inverse() {
		return re(v().inverse());
	}

	
	@Override
	public NeuronValue add(NeuronValue value) {
		if (value == null || !(value instanceof IndexedNeuronValue)) return null;
		return re(v().add(((IndexedNeuronValue)value).v()));
	}

	
	@Override
	public NeuronValue subtract(NeuronValue value) {
		if (value == null || !(value instanceof IndexedNeuronValue)) return null;
		return re(v().subtract(((IndexedNeuronValue)value).v()));
	}

	
	@Override
	public NeuronValue multiply(NeuronValue value) {
		if (value == null || !(value instanceof IndexedNeuronValue)) return null;
		return re(v().multiply(((IndexedNeuronValue)value).v()));
	}

	
	@Override
	public NeuronValue multiply(WeightValue value) {
		if (value == null || !(value instanceof IndexedWeightValue)) return null;
		return re(v().multiply(((IndexedWeightValue)value).v()));
	}

	
	@Override
	public NeuronValue multiply(double value) {
		return re(v().multiply(value));
	}

	
	@Override
	public NeuronValue multiplyDerivative(NeuronValue derivative) {
		if (derivative == null || !(derivative instanceof IndexedNeuronValue)) return null;
		return re(v().multiplyDerivative(((IndexedNeuronValue)derivative).v()));
	}

	
	@Override
	public NeuronValue divide(NeuronValue value) {
		if (value == null || !(value instanceof IndexedNeuronValue)) return null;
		return re(v().divide(((IndexedNeuronValue)value).v()));
	}

	
	@Override
	public NeuronValue divide(double value) {
		return re(v().divide(value));
	}

	
	@Override
	public NeuronValue power(double exponent) {
		return re(v().power(exponent));
	}

	
	@Override
	public NeuronValue sqrt() {
		return re(v().sqrt());
	}

	
	@Override
	public NeuronValue exp() {
		return re(v().exp());
	}


	@Override
	public NeuronValue log() {
		return re(v().log());
	}

	
	@Override
	public double mean() {
		return v().mean();
	}

	
	@Override
	public double norm() {
		return v().norm();
	}

	
	@Override
	public NeuronValue valueOf(double value) {
		return re(v().valueOf(value));
	}

	
	@Override
	public NeuronValue min(NeuronValue value) {
		if (value == null || !(value instanceof IndexedNeuronValue)) return null;
		return re(v().min(((IndexedNeuronValue)value).v()));
	}

	
	@Override
	public NeuronValue max(NeuronValue value) {
		if (value == null || !(value instanceof IndexedNeuronValue)) return null;
		return re(v().max(((IndexedNeuronValue)value).v()));
	}

	
	@Override
	public boolean matrixIsInvertible(NeuronValue[][] matrix) {
		throw new RuntimeException("Method IndexedNeuronValueV.matrixIsInvertible(NeuronValue[][]) not implemented yet");
	}


	@Override
	public NeuronValue matrixDet(NeuronValue[][] matrix) {
		throw new RuntimeException("Method IndexedNeuronValueV.matrixDet(NeuronValue[][]) not implemented yet");
	}


	@Override
	public NeuronValue[][] matrixInverse(NeuronValue[][] matrix) {
		throw new RuntimeException("Method IndexedNeuronValueV.matrixInverse(NeuronValue[][]) not implemented yet");
	}

	
	@Override
	public NeuronValue[][] matrixSqrt(NeuronValue[][] matrix) {
		throw new RuntimeException("Method IndexedNeuronValueV.matrixSqrt(NeuronValue[][]) not implemented yet");
	}

	
	@Override
	public NeuronValue[] flatten(int smallerDim) {
		throw new RuntimeException("Method IndexedNeuronValueV.flatten(int) not implemented yet");
	}

	
	@Override
	public NeuronValue[] flatten(NeuronValue[] array, int smallerDim) {
		throw new RuntimeException("Method IndexedNeuronValueV.flatten(NeuronValue[], int) not implemented yet");
	}

	
	@Override
	public NeuronValue aggregate(NeuronValue[] array) {
		throw new RuntimeException("Method IndexedNeuronValueV.aggregate(NeuronValue[]) not implemented yet");
	}

	
	@Override
	public NeuronValue[] aggregate(NeuronValue[] array, int largerDim) {
		throw new RuntimeException("Method IndexedNeuronValueV.aggregate(NeuronValue[], int) not implemented yet");
	}

	
	@Override
	public NeuronValue evaluate(Function f) {
		return re(f.evaluate(v()));
	}


	@Override
	public NeuronValue derivative(Function f) {
		return re(f.derivative(v()));
	}

	
	@Override
	public NeuronValue evaluateInverse(FunctionInvertible f) {
		return re(f.evaluateInverse(v()));
	}


	@Override
	public NeuronValue derivativeInverse(FunctionInvertible f) {
		return re(f.derivativeInverse(v()));
	}


	@Override
	public NeuronValue v() {
		return values[getIndex()];
	}

	
	/**
	 * Re-indexing and renewing the specified value. 
	 * @param value specified value.
	 * @return re-indexed value.
	 */
	private IndexedNeuronValue renew(NeuronValue value) {
		if (value == null || !(value instanceof NeuronValueV)) return null;
		IndexedNeuronValueV newValue = null;
		try {newValue = (IndexedNeuronValueV)Util.cloneBySerialize(this);}
		catch (Throwable e) {Util.trace(e);}
		
		if (newValue != null) newValue.values[getIndex()] = (NeuronValueV)value;
		return newValue;
	}
	
	
	/**
	 * Re-indexing the specified value. 
	 * @param value specified value.
	 * @return re-indexed value.
	 */
	public IndexedNeuronValue re(NeuronValue value) {
		return renew(value);
	}
	
	
	@Override
	public int getIndex() {
		return index;
	}


	@Override
	public void setIndex(int index) {
		this.index = index;
	}


	@Override
	public int size() {
		return values.length;
	}


	@Override
	public NeuronValue get(int index) {
		return values[index];
	}
	
	
	@Override
	public NeuronValue set(int index, NeuronValue value) {
		if (value == null || !(value instanceof NeuronValueV)) return null;
		NeuronValue old = get(index);
		values[index] = (NeuronValueV)value;
		return old;
	}


}
