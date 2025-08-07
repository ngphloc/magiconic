/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.core.function;

import net.ea.ann.core.value.NeuronValue;
import net.ea.ann.core.value.NeuronValueV;

/**
 * This class represents rectified linear unit (ReLU) function with vector variable.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class ReLUV implements ReLU {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;

	
	/**
	 * Minimum value.
	 */
	private double[] min = null;

	
	/**
	 * Maximum value.
	 */
	private double[] max = null;

	
	/**
	 * Constructor with minimum array and maximum array.
	 * @param min minimum array.
	 * @param max maximum array.
	 */
	public ReLUV(double[] min, double[] max) {
		this.min = min;
		this.max = max;
	}

	
	/**
	 * Constructor with minimum array.
	 * @param min minimum array.
	 */
	public ReLUV(double[] min) {
		this.min = min;
		this.max = null;
	}

	
	/**
	 * Constructor with dimension, minimum and maximum.
	 * @param dim specific dimension.
	 * @param min minimum value.
	 * @param max maximum value.
	 */
	public ReLUV(int dim, double min, double max) {
		this.min = new double[dim];
		this.max = new double[dim];
		for (int i = 0; i < dim; i++) {
			this.min[i] = min;
			this.max[i] = max;
		}
	}

	
	/**
	 * Constructor with dimension and minimum.
	 * @param dim specific dimension.
	 * @param min minimum value.
	 */
	public ReLUV(int dim, double min) {
		this.min = new double[dim];
		this.max = null;
		for (int i = 0; i < dim; i++) this.min[i] = min;
	}

	
	/**
	 * Constructor with dimension.
	 * @param dim specific dimension.
	 */
	public ReLUV(int dim) {
		this.min = new double[dim];
		this.max = new double[dim];
		for (int i = 0; i < dim; i++) {
			this.min[i] = 0;
			this.max[i] = 1;
		}
	}

	
	/**
	 * Checking whether to concern maximum.
	 * @return whether to concern maximum.
	 */
	private boolean isConcernMax() {
		return max != null && max.length > 0;
	}

	
	@Override
	public boolean isNorm() {
		if (min.length == 0 || !isConcernMax()) return false;
		for (int i = 0; i < min.length; i++) {
			if (min[i] != 0 || max[i] != 1) return false;
		}
		return true;
	}

	
	@Override
	public NeuronValue evaluate(NeuronValue x) {
		int n = max.length;
		NeuronValueV result = new NeuronValueV(n, 0.0);
		NeuronValueV v = (NeuronValueV)x;
		for (int i = 0; i < n; i++) {
			if (isConcernMax())
				result.set(i, Math.max(min[i], Math.min(max[i], v.get(i))));
			else
				result.set(i, Math.max(min[i], v.get(i)));
		}
		
		return result;
	}

	
	@Override
	public NeuronValue derivative(NeuronValue x) {
		int n = max.length;
		NeuronValueV result = new NeuronValueV(n, 0.0);
		NeuronValueV v = (NeuronValueV)x;
		for (int i = 0; i < n; i++) {
			if ((v.get(i) < min[i]) || (isConcernMax() && v.get(i) > max[i]))
				result.set(i, 0);
			else
				result.set(i, 1);
		}
		
		return result;
	}


	@Override
	public NeuronValue evaluateInverse(NeuronValue y) {
		return evaluate(y);
	}


	@Override
	public NeuronValue derivativeInverse(NeuronValue y) {
		return derivative(y);
	}


}
