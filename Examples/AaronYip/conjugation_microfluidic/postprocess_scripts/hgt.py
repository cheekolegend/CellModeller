import numpy as np
from itertools import combinations

def count_hgt_events(cells):
    """
    Sums the cumulative HGT events
    
    @param  cells           cellStates dict
    @return sum_hgt_events  total HGT events within the cellStates dict
    """
    sum_hgt_events = 0
    for cell in cells.values():
        sum_hgt_events += cell.hgt_events
        
    return sum_hgt_events      
   
def calc_hgt_efficiency(populations, recip_types=[1], trans_types=[2]):
    """
    Calculates conjugation efficiency as transconjugants per total recipient pool
    
    @param  populations total population counts
    @param  recips      recipient cellTypes
    @param  trans       transconjugant cellTypes
    @return 
    """
    recips = np.sum(populations[recip_types])
    trans = np.sum(populations[trans_types])
    hgt_efficiency = trans/(recips + trans)
    
    return hgt_efficiency
    
def count_dr_tr_contacts(G, donor_label='donor', recip_label='recip', trans_label='trans'):
    """
    Counts the number of donor/recip and trans/recip contacts in a contact graph
    @param  G       diGraph containing cell contacts for a single time point; there is 1 directed contact edge per physical contact. 
    @return         total number of donor/recip and trans/recip contacts in G
    """
    # Get number of all types of adjacencies
    cellTypes = [donor_label, recip_label, trans_label]
    adjacency_dict = get_adjacency_dict(G, cellTypes)
    
    # Count donor/recip and trans/recip contacts
    donor_recip = adjacency_dict[(donor_label, recip_label)]
    trans_recip = adjacency_dict[(recip_label, trans_label)]
      
    return donor_recip, trans_recip
         
def get_adjacency_dict(G, cellTypes):
    """
    Count the total number of conctacts between each cellType.
    The output is a single-count of physical contacts (i.e. the total counts should add up to the total number of contacts)
    @param  G               diGraph containing cell contacts for a single time point; there is 1 directed contact edge per physical contact. 
                            See read_contact_networks in contact_network.py and creation of the lineage tree in lineage_analysis.py
    @param  cellTypes       list of cellTypes
    @return adjacency_dict  dict containing total counts of adjacency types. 
    """       
    
    # Convert G to an undirected graph (no lineage edges present)
    G_copy = G.to_undirected()
    
    # Generate edge list. Note: this method double-counts edges in a non-directed graph
    edge_list = []
    for n, nbrs in G_copy.adjacency(): #n = node, nbrs = list of neighbours
        node1 = G_copy.nodes[n]['cellType']
        for nbr in nbrs: #nbr = neighbour of node n
            node2 = G_copy.nodes[nbr]['cellType']
            edge_list.append((node1, node2))
        
    # List of adjacency types
    adjacency_type = []
    pairs = list(combinations(cellTypes, 2)) # Using combinations instead of permutations avoids double-counting contacts for different cellType pairs
    for i in pairs:
        adjacency_type.append(i) #pairs of non-equal cellType
    for j in cellTypes:
        adjacency_type.append((j,j)) #pairs of same cellType
    
    #Create dict with each adjacency type
    adjacency_dict = {}
    for pair in adjacency_type:
        adjacency_dict[pair] = 0
    
    # Count edge types in the graph
    for pair in adjacency_type:
        if pair[0] == pair[1]:
            adjacency_dict[pair] += edge_list.count(pair)/2 # Avoid double-counting pairs with same cellType
        else:
            adjacency_dict[pair] += edge_list.count(pair)
    
    return adjacency_dict
