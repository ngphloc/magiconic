/**
 * AI project provide artificial intelligence solutions
 * (C) Copyright by Loc Nguyen
 * Project homepage: ai.locnguyen.net
 * Email: ng_phloc@yahoo.com
 * Phone: +84-975250362
 */
package net.hudup;

import net.ea.ann.adapter.gen.GenModelRemote;
import net.ea.ann.adapter.gen.GenModelRemoteWrapper;
import net.ea.pso.adapter.PSORemote;
import net.ea.pso.adapter.PSORemoteWrapper;
import net.hudup.core.Firer;
import net.hudup.core.alg.AlgRemote;
import net.hudup.core.alg.AlgRemoteWrapper;

/**
 * This is advanced plug-in manager which derives from {@link Firer}.
 * 
 * @author Loc Nguyen
 * @version 2.0
 *
 */
public class AIFirer extends Firer {

	
	@Override
	public void fireSimply() {
		super.fireSimply();
	}

	
	@Override
	public AlgRemoteWrapper wrap(AlgRemote remoteAlg, boolean exclusive) {
		if (remoteAlg instanceof GenModelRemote)
			return new GenModelRemoteWrapper((GenModelRemote)remoteAlg, exclusive);
		else if (remoteAlg instanceof PSORemote)
			return new PSORemoteWrapper((PSORemote)remoteAlg, exclusive);
		else
			return super.wrap(remoteAlg, exclusive);
	}

	
}
