"""
utils.py contains helper function to deal with creating founders,
processing genetic map and processing sample maps.
"""

import numpy as np
import allel
import pandas as pd
import os
import scipy.interpolate
import copy

from .person import Person, create_new, create_new_non_rec

def get_chm_info(genetic_map,vcf_data):

    """
    get chromosome length in morgans from genetic map.
    Assumes genetic_map is sorted.
    """

    # getting chm number...
    chm = list(set(vcf_data["variants/CHROM"]))
    if len(chm) == 1:
        chm = chm[0]
    else:
        raise Exception("Multiple chromosomes in this file!!!")
    chm = chm.lstrip("chr") # in some reference files we have this instead of just chr number. 22 or chr22

    # read in genetic map and subset to chm number.
    genetic_df = pd.read_csv(genetic_map,delimiter="\t",header=None,comment="#",dtype=str)
    genetic_df.columns = ["chm","pos","cM"]
    genetic_chm = genetic_df[genetic_df["chm"]==chm]

    if len(genetic_chm) == 0:
        genetic_chm = genetic_df[genetic_df["chm"]=="chr"+chm] # sometimes it is called chr22 instead of 22

    genetic_chm = genetic_chm.astype({"chm":str,"pos":int,"cM":float})

    # get length of chm.
    chm_length_morgans = max(genetic_chm["cM"])/100.0

    # get snp info - snps in the vcf file and their cm values.
    # then compute per position probability of being a breapoint.
    # requires some interpolation and finding closest positions.
    """
    # 1: Minimum in a sorted array approach and implemented inside admix().
        - O(logn) every call to admix. Note that admix() takes O(n) anyway.
    # 2: Find probabilities using span. - One time computation.

    """
    # This adds 0 overhead to code runtime.
    # get interpolated values of all reference snp positions
    genomic_intervals = scipy.interpolate.interp1d(x=genetic_chm["pos"].to_numpy(), y=genetic_chm["cM"].to_numpy(),fill_value="extrapolate")
    genomic_intervals = genomic_intervals(vcf_data["variants/POS"])
    # Method 1: a bit incorrect - to ise midpoints
    # # get their midpoints. goal is to get length spanned by each reference snp in the chromosome
    # midpts = (genomic_intervals[0:-1] + genomic_intervals[1:])/2
    # start_end_pts = np.concatenate((np.array([0]),midpts,np.array([genomic_intervals[-1]])))
    # lengths = start_end_pts[1:] - start_end_pts[0:-1]
    # # normalize the lengths after removing first since first point is not a breaking point by design
    # bp = lengths[1:] / lengths[1:].sum()
    # Method 2: simpler and more accurate
    lengths = genomic_intervals[1:] - genomic_intervals[0:-1]
    bp = lengths / lengths.sum()

    genetic_map_data = {}
    genetic_map_data["chm"] = chm
    print(f'vcf_data["calldata/GT"].shape[0] :{vcf_data["calldata/GT"].shape[0]}')
    genetic_map_data["chm_length_snps"] = vcf_data["calldata/GT"].shape[0]
    genetic_map_data["chm_length_morgans"] = chm_length_morgans
    print(f'chm_length_morgans:{chm_length_morgans}')
    genetic_map_data["breakpoint_probability"] = bp

    return genetic_map_data

def get_sample_map_data(sample_map, vcf_data, sample_weights = None):
    
    """
    Inputs:
    sample_map: tab delimited file with sample, population and no header.
    vcf_data: allel.read_vcf output. It is the reference vcf file information.
    
    Returns:
    sample_map_data: dataframe with sample, population, population_code and index in vcf_data referecnce.
    
    """
    
    # reading sample map
    #sample_map_data = pd.read_csv(sample_map,delimiter="\t",header=None,comment="#")
    #sample_map_data.columns = ["sample","population"]

    # creating ancestry map into integers from strings
    # id is based on order in sample_map file.
    #ancestry_map = {}
    #curr = 0
    #for i in sample_map_data["population"]:
        #if i in ancestry_map.keys():
            #continue
        #else:
            #ancestry_map[i] = curr
            #curr += 1
    # print("Ancestry map",ancestry_map)
    sample_map_data["population_code"] = np.vectorize(ancestry_map.get)(sample_map_data["population"])

    # getting index of samples in the reference files

    b = vcf_data["samples"]
    a = np.array(list(sample_map_data["sample"]))

    sorter = np.argsort(b)
    indices = sorter[np.searchsorted(b, a, sorter=sorter)]
    sample_map_data["index_in_reference"] = indices
    
    if sample_weights is not None:
        sample_weights_df = pd.read_csv(sample_weights,delimiter="\t",header=None,comment="#")
        sample_weights_df.columns = ["sample","sample_weight"]
        sample_map_data = pd.merge(sample_map_data, sample_weights_df, on='sample')

    else:
        sample_map_data["sample_weight"] = [1.0/len(sample_map_data)]*len(sample_map_data)
    
    return sample_map_data

def build_founders(vcf_data, genetic_map_data, sample_map_data, sample_weight=None):
    
    """
    Returns founders - a list of Person datatype.
    founders_weight - a list with a weight for each sample in founders

    Inputs
    genetic_map_data - output of get_chm_info
    
    """

    # information about snps, chm, lengths from reference, genetic map.
    chm = genetic_map_data["chm"]
    chm_length_morgans = genetic_map_data["chm_length_morgans"]
    chm_length_snps = genetic_map_data["chm_length_snps"]

    # building founders
    founders = []
    founders_idx = {}

    for i in sample_map_data.iterrows():

        # first get the index of this sample in the vcf_data.
        # if not there, skip and print to log.

        index = i[1]["ref_idx"]
        founders_idx[2*index]=i[0]
        founders_idx[(2*index)+1]=i[0]

        name = i[1]["Sample"]

        # when creating maternal, paternal make sure it has same keys


        maternal = {}
        paternal = {}

        # let us use the first for maternal in the vcf file...
        maternal["snps"] = vcf_data["calldata/GT"][:,index,0]
        paternal["snps"] = vcf_data["calldata/GT"][:,index,1]

        # single ancestry assumption.
        maternal["superpop"] = np.array([i[1]["superpop"]]*chm_length_snps)
        paternal["superpop"] = np.array([i[1]["superpop"]]*chm_length_snps)
        maternal["granular_pop"] = np.array([i[1]["granular_pop"]]*chm_length_snps)
        paternal["granular_pop"] = np.array([i[1]["granular_pop"]]*chm_length_snps)
        maternal["vcf_idx"] = np.array([2*index for _ in range(chm_length_snps)])
        paternal["vcf_idx"] = np.array([(2*index)+1 for _ in range(chm_length_snps)])
        
        # any more info like coordinates, prs can be added here.

        p = Person(chm,chm_length_morgans,chm_length_snps,maternal,paternal,name)

        founders.append(p)
        
    if sample_weight is not None:
        founders_weight = sample_map_data["sample_weight"]

        return founders,founders_weight,founders_idx
    else:
        return founders,founders_idx

def getFoundersForCombineAdmix(prevAdmixedSample, originalFounders, foundersIdx):
    """
    returns the founders to be used for each admixture cycle/ admixture sample
    """

    # from the list of founders objects (Person instances), pick those instances where
    # the founder object(Person instance) are equal
    founders=[]
    vcfIdxLs=list(np.unique(prevAdmixedSample.maternal['vcf_idx']))+ list(np.unique(prevAdmixedSample.paternal['vcf_idx']))
    
    for v in vcfIdxLs:
        idx=foundersIdx[v]
        founders.append(originalFounders[idx])
    return founders

def create_dataset(founders,num_samples_per_gen,gens_to_ret,breakpoint_probability=None,random_seed=42,\
    verbose=True, founders_weight=None):
    
    np.random.seed(random_seed)

    if breakpoint_probability is None:
        print("Warning! Genetic map not taken into account when sampling breakpoints.")
        print("Uniformly sampling from given snps.")
    
    max_gen = max(gens_to_ret)
    
    overall = {}
    overall[0] = founders

    # could be greater than or equal to 100 to avoid founder bias.
    number_of_admixed_samples = max(400,num_samples_per_gen)

    for gen in range(1,max_gen+1):
        if verbose:
            print("Simulating generation ",gen)
        this_gen_people = []


        for i in range(number_of_admixed_samples):
            # select any 2 parents from prev. gen.
            # if 1st generation, choose from founders
            if gen == 1:
                if founders_weight is not None:
                    p = founders_weight
                else:
                    p = 1/len(founders)
                    
                id1,id2 = np.random.choice(len(founders),size=2,replace=False, p=p)
                 
            # else choose from num_samples_per_gen
            else:
                id1,id2 = np.random.choice(number_of_admixed_samples,size=2,replace=False)
            p1 = overall[gen-1][id1]
            p2 = overall[gen-1][id2]

            adm = create_new(p1,p2,breakpoint_probability)
            this_gen_people.append(adm)

        overall[gen] = this_gen_people

        if gen-1 not in gens_to_ret and gen-1 !=0:
            del overall[gen-1]

    # for every requested generation, only keep num_samples_per_gen
    for key in overall.keys():
        if key == 0:
            continue
        overall[key] = overall[key][0:num_samples_per_gen]
        # overall[key] = list(np.random.choice(overall[key], \
        #                            size=num_samples_per_gen,replace=False))

    return overall

def create_non_rec_dataset(founders,num_samples_per_gen,gens_to_ret,breakpoint_probability=None,random_seed=42,verbose=True, \
    founders_weight=None, prevAdmixedFlag=False, prevAdmixed=None, foundersIdx=None):
    
    np.random.seed(random_seed)
    print(f"prevAdmixedFlag:{prevAdmixedFlag}")
    if prevAdmixedFlag:
        assert (prevAdmixed is not None) and (foundersIdx is not None), "previous admixed flag is true but previous admixed\
            sample data was not provided"
        founders_copy=copy.deepcopy(founders)

    if breakpoint_probability is None:
        print("genetic map not taken into account when sampling breakpoints.")
        print("Uniformly sampling from given snps.")
    
    max_gen = max(gens_to_ret)
    
    overall = {}
    overall[0] = founders

    # could be greater than or equal to 100 to avoid founder bias.
    number_of_admixed_samples = num_samples_per_gen
    select_idx = {}
    for gen in gens_to_ret:
        select_idx[gen]=[]
        # BUG fix: Gen ==0 means do nothing.
        if gen == 0:
            continue

        if verbose:
            print("Simulating generation ",gen)
        this_gen_people = []
        
        for i in range(number_of_admixed_samples):
            # additional logic for using previous admixed sample founders
            if prevAdmixedFlag:
                prevAdmixedSample = prevAdmixed[gen][i]
                founders=getFoundersForCombineAdmix(prevAdmixedSample, founders_copy, foundersIdx)
            # use the new admixture tool for non-recursive simulation
            if founders_weight is None:
                founders_weight = [1/len(founders) for _ in range(len(founders))]
            adm, m_idx, p_idx = create_new_non_rec(founders,founders_weight,gen,breakpoint_probability,prevAdmixedFlag)
            this_gen_people.append(adm)
            select_idx[gen].append([m_idx,p_idx])

        overall[gen] = this_gen_people

    return overall, select_idx

def write_output(root,dataset,verbose=True):
    
    """
    creates output numpy files in root directory - under that, we will have gen_{}
    folders and then we have the npy files.
    Make sure root has chm number in it.
    
    """

    if not os.path.isdir(root):
        os.makedirs(root)

    for gen in dataset.keys():
        
        if verbose:
            print("Writing generation: {}".format(gen))

        if gen not in dataset.keys():
            if verbose:
                print("Did not simulate gen {}".format(gen))
            continue

        # make directory
        gen_dir = os.path.join(root, "gen_{}".format(gen))
        if not os.path.isdir(gen_dir):
            os.makedirs(gen_dir)

        snps = []
        vcf_idx = []
        for person in dataset[gen]:
            snps.append(person.maternal["snps"])
            snps.append(person.paternal["snps"])
            vcf_idx.append(person.maternal["vcf_idx"])
            vcf_idx.append(person.paternal["vcf_idx"])

        # create npy files.
        snps = np.stack(snps)
        np.save(gen_dir+"/mat_vcf_2d.npy",snps)

        # create map files.
        vcf_idx = np.stack(vcf_idx)
        np.save(gen_dir+"/mat_map.npy",vcf_idx)
