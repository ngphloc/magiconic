/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.core;

import java.io.Serializable;

/**
 * This interface represents information event about neural network.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public interface NetworkInfoEvent extends Serializable, Cloneable {

	
	/**
	 * Getting information.
	 * @return information.
	 */
	String getInfo();
	
	
	/**
	 * Setting information.
	 * @param info specified information.
	 */
	void setInfo(String info);
	
	
}
