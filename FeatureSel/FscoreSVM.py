from mvpa2.suite import *
from mvpa2.clfs import *
from mvpa2.testing.datasets import datasets

"""Load"""


path = os.path.join(pymvpa_datadbroot)

ds = datasets['uni4large']

#normalize
zscore(ds)

print ds.shape
#Based on Fscore

clf = LinearCSVMC()

fsel = SensitivityBasedFeatureSelection(
            OneWayAnova(),
	    #keep best 5%
            #FractionTailSelector(0.05, mode='select', tail='upper'))
	    #or a fix number of feature
	    FixedNElementTailSelector(4, mode='select', tail='upper'))
    
fclf = FeatureSelectionClassifier(clf,fsel)

cvte = CrossValidation(fclf, NFoldPartitioner(),
                        enable_ca=['stats'])
fsel.train(ds)

ds_p = fsel(ds)
results = cvte(ds_p)


print ds_p.shape
print cvte.ca.stats.as_string(description=True)

