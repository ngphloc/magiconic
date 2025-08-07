/**
 * AI: Artificial Intelligent Project
 * (C) Copyright by Loc Nguyen's Academic Network
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.ea.ann.adapter.gen.beans;

import net.ea.ann.adapter.gen.GenModelAbstract;
import net.ea.ann.core.Util;
import net.ea.ann.gen.ConvGenModel;
import net.ea.ann.gen.pixel.PixelRNNImpl;

/**
 * This class is the bean implementation of pixel recurrent neural network for Hudup framework.
 * 
 * @author Loc Nguyen
 * @version 1.0
 *
 */
public class PixelRNN extends GenModelAbstract {


	/**
	 * Serial version UID for serializable class.
	 */
	private static final long serialVersionUID = 1L;


	/**
	 * Default constructor.
	 */
	public PixelRNN() {

	}

	
	@Override
	public String getName() {
		String name = getConfig().getAsString(DUPLICATED_ALG_NAME_FIELD);
		if (name != null && !name.isEmpty())
			return name;
		else
			return "pixelrnn";
	}

	
	@Override
	protected ConvGenModel createGenModel() {
		try {
			ConvGenModel model = PixelRNNImpl.create(getNeuronChannel(), getRasterChannel());
			if (model instanceof net.ea.ann.gen.GenModelAbstract)
				((net.ea.ann.gen.GenModelAbstract)model).setNorm(isNorm());
			return model;
		} catch (Throwable e) {Util.trace(e);}
		return null;
	}

	
}
