/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.conv;

import net.ea.ann.core.Neuron;
import net.ea.ann.core.value.NeuronValue;

/**
 * This interface represents a convolutional neuron.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface ConvNeuron extends Neuron {

	
	/**
	 * Setting value.
	 * @param value specific value.
	 * @return previous value.
	 */
	NeuronValue setValue(NeuronValue value);
	
	
	/**
	 * Getting input.
	 * @return input.
	 */
	NeuronValue getInput();
	
	
	/**
	 * Setting input.
	 * @param input input.
	 */
	void setInput(NeuronValue input);
	
	
	/**
	 * Getting tag.
	 * @return tag.
	 */
	Object getTag();
	
	
	/**
	 * Setting tag.
	 * @param tag tag.
	 */
	void setTag(Object tag);

	
	/**
	 * Clearing neuron.
	 */
	void clear();
	
	
}
