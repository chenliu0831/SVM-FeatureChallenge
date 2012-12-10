from mvpa2.suite import *
from mvpa2.clfs import *
from mvpa2.testing.datasets import datasets

"""Load"""


path = os.path.join(pymvpa_datadbroot)

ds = normal_feature_dataset(perlabel=20, nlabels=2,
                            nfeatures=30, nonbogus_features=[1,5],
                             snr=1.)

#Based on Fscore

#cvte = CrossValidation(clf, NFoldPartitioner(),
	# errorfx=lambda p, t: np.mean(p == t))
#clf = LinearCSVMC()
#cvte = CrossValidation(clf, NFoldPartitioner(),
#                        enable_ca=['stats'])


#fsel = SensitivityBasedFeatureSelection(
#            OneWayAnova(),
#            FixedNElementTailSelector(10, mode='select', tail='upper'))

#fsel.train(ds)

#ds_fsel = fsel(ds)


#rfe

ds = datasets['uni4large']


zscore(ds)
percent = 80
rfesvm_split = SplitClassifier(LinearCSVMC())
fs = \
	RFE(rfesvm_split.get_sensitivity_analyzer(
      	postproc=ChainMapper([
                #FxMapper('features', l2_normed),
                #FxMapper('samples', np.mean),
                #FxMapper('samples', np.abs)
                FxMapper('features', lambda x: np.argsort(np.abs(x))),
                #maxofabs_sample()
                mean_sample()
                ])),	
                ProxyMeasure(rfesvm_split,
                             postproc=BinaryFxNode(mean_mismatch_error,
                                                   'targets')),
                Splitter('train'),
                fselector=FractionTailSelector(
                    percent / 100.0,
                    mode='select', tail='upper'), update_sensitivity=True)
clf = FeatureSelectionClassifier(
            LinearCSVMC(),
            # on features selected via RFE
            fs)
             # update sensitivity at each step (since we're not using the
             # same CLF as sensitivity analyzer)
cv = CrossValidation(clf, NFoldPartitioner(), postproc=mean_sample(),
                             enable_ca=['stats','confusion'])

error = cv(ds).samples.squeeze()

print error

print cv.ca.stats.as_string(description=True)
