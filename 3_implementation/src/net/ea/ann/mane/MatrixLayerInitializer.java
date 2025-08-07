/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.mane;

import java.awt.Dimension;
import java.io.Serializable;

import net.ea.ann.conv.filter.Filter2D;

/**
 * This utility class provides initialization methods for matrix network layer.
 *  
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class MatrixLayerInitializer implements Cloneable, Serializable {


	/**
	 * Serial version UID for serializable class. 
	 */
	private static final long serialVersionUID = 1L;
	
	
	/**
	 * Internal matrix network layer.
	 */
	protected MatrixLayerImpl layer = null;
	

	/**
	 * Constructor with layer.
	 * @param layer layer.
	 */
	public MatrixLayerInitializer(MatrixLayerImpl layer) {
		this.layer = layer;
	}
	

	/**
	 * Initializing layer with size, previous layer size, and filter.
	 * @param size this size.
	 * @param prevSize previous layer size. It can be null.
	 * @param filter filter. It can be null.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Dimension size, Dimension prevSize, Filter2D filter) {
		return layer.initialize(size, prevSize, filter);
	}

	
	/**
	 * Initializing layer with size, previous layer, and filter.
	 * @param size this size.
	 * @param prevSize previous size.
	 * @param prevLayer previous layer. It can be null.
	 * @param filter filter. It can be null.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Dimension size, Dimension prevSize, MatrixLayerAbstract prevLayer, Filter2D filter) {
		return layer.initialize(size, prevSize, prevLayer, filter);
	}
	
	
	/**
	 * Initializing layer with size and previous layer.
	 * @param size this size.
	 * @param prevSize previous size.
	 * @param prevLayer previous layer. It can be null.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Dimension size, Dimension prevSize, MatrixLayerAbstract prevLayer) {
		return initialize(size, prevSize, prevLayer, null);
	}
	
	
	/**
	 * Initializing layer with size and previous layer size.
	 * @param size this size.
	 * @param prevSize previous layer size. It can be null.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Dimension size, Dimension prevSize) {
		return initialize(size, prevSize, (Filter2D)null);
	}
	
	
	/**
	 * Initializing layer with size and filter.
	 * @param size this size.
	 * @param filter filter. It can be null.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Dimension size, Filter2D filter) {
		return initialize(size, (Dimension)null, filter);
	}

	
	/**
	 * Initializing layer with size.
	 * @param size this size.
	 * @return true if initialization is successful.
	 */
	public boolean initialize(Dimension size) {
		return initialize(size, (Dimension)null);
	}

	
}
