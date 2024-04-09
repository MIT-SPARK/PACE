import sys, os
import copy
import time

import cvxpy as cp
import numpy as np
from itertools import combinations, product
from collections import defaultdict
import networkx as nx
import scipy as scipy

script_dir = os.path.realpath(os.path.dirname(__file__))
sys.path.append(os.path.join(script_dir, "../../lib/python/"))
import debug_utils
from debug_utils import debug_print
from sdp_data import get_rotation_relaxation_constraints, get_vectorization_permutation
import robin_py

ORDER_NOT_COVISIBLE = -1
ORDER_IJK = 1
ORDER_IKJ = 2
ORDER_BOTH = 3
ORDER_UNKNOWN = 4


def compute_lam(lam_base, N, K):
    return lam_base * np.sqrt(float(K) / float(N))


def inlier_stats(A, B):
    '''
    A contains the indices of ground-truth inliers
    B contains the indices of robin estimated inliers
    '''
    common = np.intersect1d(A, B)
    if B.shape[0] > 0:
        B_inlier_rate = float(common.shape[0]) / float(B.shape[0])
    else:
        B_inlier_rate = 0

    num_A_and_B = common.shape[0]
    num_A_not_in_B = A.shape[0] - common.shape[0]
    num_B_not_in_A = B.shape[0] - common.shape[0]

    return np.array([B_inlier_rate, num_A_and_B, num_A_not_in_B, num_B_not_in_A])


def rotation_error(R0, R1):
    return np.abs(
        np.arccos(np.clip((np.trace(R0.T @ R1) - 1) / 2.0, -0.999999,
                          0.999999))) / np.pi * 180


def translation_error(t0, t1):
    return np.linalg.norm(t0 - t1)


def project_to_SO3(A):
    U, S, Vh = np.linalg.svd(A)
    R = U @ Vh
    if np.linalg.det(R) < 0:
        R = U @ np.diag([1, 1, -1]) @ Vh

    # RRtran = R @ R.T 
    # assert(
    #     np.linalg.norm(RRtran-np.identity(3),ord='fro')<1e-12,'Projection to SO3 failed')
    return R


def decompose_nonrigid_model(A):
    '''
    If A is the model for a nonrigid registration problem, then decompose A into 
    9 basis shapes such that the affine transformation applied on A is equivalent
    to a linear combination of the 9 basis shapes
    '''
    N = A.shape[1]
    zero = np.zeros((1, N))
    row1 = [A[0, :]]
    row2 = [A[1, :]]
    row3 = [A[2, :]]
    A1 = np.concatenate((
        row1, zero, zero), axis=0)
    A2 = np.concatenate((
        zero, row1, zero), axis=0)
    A3 = np.concatenate((
        zero, zero, row1), axis=0)
    A4 = np.concatenate((
        row2, zero, zero), axis=0)
    A5 = np.concatenate((
        zero, row2, zero), axis=0)
    A6 = np.concatenate((
        zero, zero, row2), axis=0)
    A7 = np.concatenate((
        row3, zero, zero), axis=0)
    A8 = np.concatenate((
        zero, row3, zero), axis=0)
    A9 = np.concatenate((
        zero, zero, row3), axis=0)

    A_basis = np.asarray([A1, A2, A3, A4, A5, A6, A7, A8, A9])

    return A_basis


def basis_to_cads(A_basis):
    K = A_basis.shape[0]
    cad_db = []
    for i in range(K):
        cad = {'kpts': np.squeeze(A_basis[i, :, :])}
        cad_db.append(cad)
    return cad_db


def minimum_distance_to_convex_hull(A):
    '''
    A is shape 3 by K, compute the minimum distance from the origin to the convex hull of A
    '''
    K = A.shape[1]
    P = A.T @ A
    one = np.ones((K, 1))
    # Use CVXPY to solve
    x = cp.Variable(K)
    prob = cp.Problem(cp.Minimize(cp.quad_form(x, P)),
                      [x >= 0,
                       one.T @ x == 1])
    prob.solve(solver='ECOS', verbose=False)
    x_val = x.value
    min_distance = np.linalg.norm(A @ x_val)
    return min_distance


def compute_min_max_distances(cad_kpts):
    print('Computing upper and lower bounds in cad pairwise distances...')

    K = cad_kpts.shape[0]
    N = cad_kpts.shape[2]
    si, sj = np.meshgrid(np.arange(N), np.arange(N))
    mask_uppertri = (sj > si)
    si = si[mask_uppertri]
    sj = sj[mask_uppertri]

    cad_TIMs_ij = cad_kpts[:, :, sj] - cad_kpts[:, :, si]  # shape K by 3 by (n-1)_tri

    # compute max distances
    cad_dist_k_ij = np.linalg.norm(cad_TIMs_ij, axis=1)  # shape K by (n-1)_tri
    cad_dist_max_ij = np.max(cad_dist_k_ij, axis=0)

    # compute min distances
    cad_dist_min_ij = []
    num_pairs = cad_TIMs_ij.shape[2]
    one_tenth = num_pairs / 10
    for i in range(num_pairs):
        tmp = cad_TIMs_ij[:, :, i].T
        min_dist = minimum_distance_to_convex_hull(tmp)
        cad_dist_min_ij.append(min_dist)
        if i % one_tenth == 1:
            print(f'{i}/{num_pairs}.')

    cad_dist_min_ij = np.array(cad_dist_min_ij)

    print('Done')

    return cad_dist_min_ij, cad_dist_max_ij


def compute_min_max_distances_with_idx_maps(cad_kpts):
    print('Computing upper and lower bounds in cad pairwise distances...')

    K = cad_kpts.shape[0]
    N = cad_kpts.shape[2]
    si, sj = np.meshgrid(np.arange(N), np.arange(N))
    mask_uppertri = (sj > si)
    si = si[mask_uppertri]
    sj = sj[mask_uppertri]

    cad_TIMs_ij = cad_kpts[:, :, sj] - cad_kpts[:, :, si]  # shape K by 3 by (n-1)_tri

    # compute max distances
    cad_dist_k_ij = np.linalg.norm(cad_TIMs_ij, axis=1)  # shape K by (n-1)_tri
    cad_dist_max_ij = np.max(cad_dist_k_ij, axis=0)

    # compute min distances
    cad_dist_min_ij = []
    num_pairs = cad_TIMs_ij.shape[2]
    one_tenth = num_pairs / 10
    for i in range(num_pairs):
        tmp = cad_TIMs_ij[:, :, i].T
        min_dist = minimum_distance_to_convex_hull(tmp)
        cad_dist_min_ij.append(min_dist)
        if i % one_tenth == 1:
            print(f'{i}/{num_pairs}.')

    cad_dist_min_ij = np.array(cad_dist_min_ij)

    print('Done')

    return cad_dist_min_ij, cad_dist_max_ij, si, sj


def measure_winding_order(sorted_triplet_points, chirality='left-handed'):
    """ Measure the winding order of given triplets (assuming i, j, k order of the ids with i<j<k)
    """
    v1 = sorted_triplet_points[:, 1] - sorted_triplet_points[:, 0]
    v1 /= np.linalg.norm(v1)
    v2 = sorted_triplet_points[:, 2] - sorted_triplet_points[:, 0]
    v2 /= np.linalg.norm(v2)
    mat = np.vstack((v1, v2)).T
    # note: the opencv coordinate system is left handed
    # ijk: determinant negative
    # ikj: determinant positive
    det = np.linalg.det(mat)
    measured_order = 0
    if chirality == 'left-handed':
        if det < 0:
            measured_order = ORDER_IJK
        elif det > 0:
            measured_order = ORDER_IKJ
        else:
            # det == 0
            # Either: collinear points, or some points share the same coordinates
            measured_order = ORDER_BOTH
    elif chirality == 'right-handed':
        if det < 0:
            measured_order = ORDER_IKJ
        elif det > 0:
            measured_order = ORDER_IJK
        else:
            # det == 0
            # Either: collinear points, or some points share the same coordinates
            measured_order = ORDER_BOTH

    return measured_order


def test_triplet_winding_order(triplet_points, triplet_semantic_ids, order_db, chirality='left-handed'):
    """Calculate the 2D winding order and compare with the order database
    Assume triplet_points are 2-by-3,
    """
    sort_index = np.argsort(list(triplet_semantic_ids))
    sorted_triplet_semantic_ids = tuple(np.array(triplet_semantic_ids)[sort_index])
    sorted_triplet_points = triplet_points[:, sort_index]
    expected_order = order_db[sorted_triplet_semantic_ids]
    # print(f"triplet points: {sorted_triplet_semantic_ids}")
    if expected_order == ORDER_BOTH:
        # print(f"both possible!")
        return True
    if expected_order == ORDER_NOT_COVISIBLE:
        # print(f"not covisible!")
        return False
    v1 = sorted_triplet_points[:, 1] - sorted_triplet_points[:, 0]
    v1 /= np.linalg.norm(v1)
    v2 = sorted_triplet_points[:, 2] - sorted_triplet_points[:, 0]
    v2 /= np.linalg.norm(v2)
    if np.isnan(v2[0]) or np.isnan(v2[1]) or np.isnan(v1[0]) or np.isnan(v1[1]):
        # points share the same coordinates
        debug_utils.debug_print("Overlapping points.")
        return True
    mat = np.vstack((v1, v2)).T
    # note: the opencv coordinate system is left handed
    # ijk: determinant negative
    # ikj: determinant positive
    det = np.linalg.det(mat)
    measured_order = 0
    if chirality == 'left-handed':
        if det < 0:
            measured_order = ORDER_IJK
            # print(f"measured order: {measured_order}")
        elif det > 0:
            measured_order = ORDER_IKJ
            # print(f"measured order: {measured_order}")
        else:
            # det == 0
            # Either: collinear points, or some points share the same coordinates
            # in this case, the order can be either, and we always add edges (hence always return True)
            # print(f"collinear")
            return True
    elif chirality == 'right-handed':
        if det < 0:
            measured_order = ORDER_IKJ
            # print(f"measured order: {measured_order}")
        elif det > 0:
            measured_order = ORDER_IJK
            # print(f"measured order: {measured_order}")
        else:
            # det == 0
            # Either: collinear points, or some points share the same coordinates
            # in this case, the order can be either, and we always add edges (hence always return True)
            # print(f"collinear")
            return True

    return measured_order == expected_order


def robin_is_clique_self_supporting(tgt, clique_semantic_ids, order_db, chirality='left-handed'):
    """ Check to see whether the clique is self-supporting:
    every triplet subset within the clique has measured order consistent with
    """
    clique_size = len(clique_semantic_ids)
    all_triplets_in_clique = combinations(list(range(clique_size)), 3)

    # iterate through all the possible triplets
    triplet_consistencies = []
    inconsistencies_count = 0
    self_supporting = True
    for index_triplet in all_triplets_in_clique:
        # convert triplet indices to semantic ids
        triplet_semantic_ids = (
            clique_semantic_ids[index_triplet[0]],
            clique_semantic_ids[index_triplet[1]],
            clique_semantic_ids[index_triplet[2]])
        triplet_points = np.vstack(
            (tgt[:, triplet_semantic_ids[0]],
             tgt[:, triplet_semantic_ids[1]],
             tgt[:, triplet_semantic_ids[2]])).T
        consistent = test_triplet_winding_order(triplet_points, triplet_semantic_ids, order_db, chirality=chirality)
        triplet_consistencies.append([triplet_semantic_ids, consistent])
        if not consistent:
            inconsistencies_count += 1
            self_supporting = False

    print(f"clique {clique_semantic_ids} self consistencies:")
    for c in triplet_consistencies:
        print(c)

    return self_supporting, inconsistencies_count


def robin_find_single_maximum_hyperclique(hypergraph, subgraph_semantic_nodes):
    """Given a hypergraph, find a maximum hyperclique

    hypergraph: a dictionary containing all triplets for the entire graph. Value is true if the edge exists.
    subgraph_nodes: if not None, then only solver for maximum hyperclique in this subgraph
    """
    # Currently the passed hypergraph is not reduced
    nodes_count = len(subgraph_semantic_nodes)
    all_triplets = combinations(list(range(nodes_count)), 3)

    # find the maximum hyperclique
    # mix integer linear program formulation:
    #
    # max sum(x_i); i=1..N
    # s.t. x_i + x_j + x_k <= 2 for all (i,j,k) not in E
    #      x_i integer variables
    x = cp.Variable(nodes_count, boolean=True)
    obj = cp.Maximize(cp.sum(x))
    constraints = []
    for triplet in all_triplets:
        triplet_semantic_ids = (
            subgraph_semantic_nodes[triplet[0]], subgraph_semantic_nodes[triplet[1]],
            subgraph_semantic_nodes[triplet[2]])
        sorted_semantic_ids = tuple(sorted(triplet_semantic_ids))
        # x_i + x_j + x_k <= 2 for all (i,j,k) not in E
        if not hypergraph[sorted_semantic_ids]:
            # adding the constraint only the triplet semantic ids do not form an hyperedge
            # note: the constraints are over the non semantic id triplet ids
            constraints.append(x[triplet[0]] + x[triplet[1]] + x[triplet[2]] <= 2)

    if len(constraints) == 0:
        # the entire (sub)graph is a hyperclique
        return [i for i in range(nodes_count)], copy.deepcopy(subgraph_semantic_nodes)

    # solve
    prob = cp.Problem(obj, constraints)
    prob.solve()

    clique_indices = [i for i in range(nodes_count) if x.value[i] == 1]
    semantic_inlier_indices = [subgraph_semantic_nodes[i] for i in clique_indices]

    return clique_indices, semantic_inlier_indices


def robin_find_all_maximum_hypercliques(hypergraph, g_nx, tgt_semantic_id_map):
    """ Start from the list of maximal simple cliques,
    run the maximum clique algorithm on the formulated subgraphs
    """
    # start from simple graph maximal cliques
    # solve the set cover problems for each maximal clique
    all_maximal_cliques = list(nx.find_cliques(g_nx))
    all_maximal_cliques = [sorted(c) for c in all_maximal_cliques]

    current_maximum_hyperclique_size = -1
    maximum_hypercliques = []
    for maximal_clique in all_maximal_cliques:
        debug_utils.debug_print(f"Current maximal clique: {maximal_clique}")
        if len(maximal_clique) < current_maximum_hyperclique_size:
            continue
        clique_semantic_ids = [tgt_semantic_id_map[x] for x in maximal_clique]

        # call the maximum hyperclique function
        # c_maximum_hyperclique_indices are the indices of the nodes belonging to the maximuim hyperclique within the
        # current maximal clique
        c_maximum_hyperclique_indices, semantic_hyperclique_ids = robin_find_single_maximum_hyperclique(hypergraph,
                                                                                                        clique_semantic_ids)
        if len(semantic_hyperclique_ids) > current_maximum_hyperclique_size:
            current_maximum_hyperclique_size = len(semantic_hyperclique_ids)
            maximum_hypercliques.append(sorted([maximal_clique[i] for i in c_maximum_hyperclique_indices]))

    # filter out non maximum hypercliques
    max_hyperclique_number = np.max([len(hc) for hc in maximum_hypercliques])
    max_hyperclique_indices = list(
        np.argwhere(np.array([len(hc) for hc in maximum_hypercliques]) == max_hyperclique_number).flatten())
    maximum_hypercliques = [x for ii, x in enumerate(maximum_hypercliques) if ii in max_hyperclique_indices]

    # assert maximum hypercliques are of the same size
    assert (len({len(i) for i in maximum_hypercliques}) == 1)
    max_hypercliques_union = np.unique(np.array(maximum_hypercliques).flatten())

    max_clique_union_indices = list(max_hypercliques_union)
    semantic_inlier_indices = [tgt_semantic_id_map[i] for i in max_clique_union_indices]
    return max_clique_union_indices, semantic_inlier_indices


def robin_find_all_maximum_hypercliques_set_cover(hypergraph, g_nx, tgt_semantic_id_map):
    """Given a hypergraph, find all maximum hypercliques

    hypergraph: dictionary of hyperedges
    """
    # start from simple graph maximal cliques
    # solve the set cover problems for each maximal clique
    all_maximal_cliques = list(nx.find_cliques(g_nx))
    all_maximal_cliques = [sorted(c) for c in all_maximal_cliques]

    current_maximum_hyperclique_size = -1
    maximum_hypercliques = []
    for maximal_clique in all_maximal_cliques:
        if len(maximal_clique) < current_maximum_hyperclique_size:
            continue
        # get the failed vertices & edges
        failed_vertices = []
        failed_edges = []
        vertex_edge_subsets = defaultdict(list)
        maximal_clique_triplets = combinations(maximal_clique, 3)
        for mc_triplet in maximal_clique_triplets:
            c_consistent = hypergraph[tuple(sorted(mc_triplet))]
            if not c_consistent:
                failed_edges.append(list(sorted(mc_triplet)))
                for v in mc_triplet:
                    vertex_edge_subsets[v].append(len(failed_edges) - 1)
                failed_vertices.extend(list(mc_triplet))
        failed_vertices = list(sorted(set(failed_vertices)))
        failed_vertices_id2idx = {}
        for idx, v in enumerate(failed_vertices):
            failed_vertices_id2idx[v] = idx

        if len(failed_vertices) == 0:
            hyperclique = maximal_clique
        else:
            # create the set cover problem
            # sets: edges that fail the consistency test
            # subsets: # of vertices in the failed edges; subset contains number of edges the vertex stay in
            # x: 1 if that subset is selected, 0 otherwise
            x = cp.Variable(len(failed_vertices), boolean=True)
            obj = cp.Minimize(cp.sum(x))
            constraints = []
            for c_verts in failed_edges:
                ids = [failed_vertices_id2idx[v] for v in c_verts]
                constraints.append(cp.sum(x[ids]) >= 1)

            # solve
            prob = cp.Problem(obj, constraints)
            prob.solve()

            # remove selected vertices
            vertex_to_remove = [failed_vertices[i] for i in range(len(failed_vertices)) if x.value[i] == 1]
            hyperclique = [v for v in maximal_clique if v not in vertex_to_remove]
            maximum_hypercliques.append(hyperclique)

    # filter out non maximum hypercliques
    max_hyperclique_number = np.max([len(hc) for hc in maximum_hypercliques])
    max_hyperclique_indices = list(
        np.argwhere(np.array([len(hc) for hc in maximum_hypercliques]) == max_hyperclique_number).flatten())
    maximum_hypercliques = [x for ii, x in enumerate(maximum_hypercliques) if ii in max_hyperclique_indices]

    # assert maximum hypercliques are of the same size
    assert (len({len(i) for i in maximum_hypercliques}) == 1)
    max_hypercliques_union = np.unique(np.array(maximum_hypercliques).flatten())

    max_clique_union_indices = list(max_hypercliques_union)
    semantic_inlier_indices = [tgt_semantic_id_map[i] for i in max_clique_union_indices]
    return max_clique_union_indices, semantic_inlier_indices


def robin_prune_outliers_triplet_hypergraph(tgt, tgt_semantic_id_map, order_db, method='single-maxclique',
                                            chirality='left-handed'):
    """Prune outliers with 3-uniform hypergraphs built from triplet order invariants

    method:
        single-maxclique: return a single maximum hyperclique
        maxclique-union: return the union of the maximum hypercliques

    :return:
    """
    assert (tgt.shape[0] == 2)
    assert (tgt.shape[1] == len(tgt_semantic_id_map))
    num_keypoints = tgt.shape[1]
    all_triplets = combinations(list(range(num_keypoints)), 3)

    # build the hypergraph
    # hypergraph is a dictionary with triplets as keys and True/False as has edge / no edge
    # note: hypergraph nodes are semantic ids, not indices
    hypergraph = {}
    # also create a graph in networkx
    # note: network x graphs are indices not semantic ids
    g_nx = nx.Graph()
    for i in range(num_keypoints):
        g_nx.add_node(i)

    for triplet in all_triplets:
        # convert triplet indices to semantic ids
        triplet_semantic_ids = (
            tgt_semantic_id_map[triplet[0]], tgt_semantic_id_map[triplet[1]], tgt_semantic_id_map[triplet[2]])
        triplet_points = np.vstack((tgt[:, triplet[0]], tgt[:, triplet[1]], tgt[:, triplet[2]])).T
        is_consistent = test_triplet_winding_order(triplet_points, triplet_semantic_ids, order_db,
                                                   chirality=chirality)
        # populate the hypergraph
        sorted_semantic_ids = tuple(sorted(triplet_semantic_ids))
        hypergraph[sorted_semantic_ids] = is_consistent
        if is_consistent:
            # add edges to nx graph
            g_nx.add_edge(triplet[0], triplet[1])
            g_nx.add_edge(triplet[0], triplet[2])
            g_nx.add_edge(triplet[1], triplet[2])

    if method == "single-maxclique":
        robin_inlier_indices, semantic_inlier_indices = robin_find_single_maximum_hyperclique(hypergraph,
                                                                                              tgt_semantic_id_map)

        return semantic_inlier_indices, robin_inlier_indices, hypergraph, g_nx
    elif method == "maxclique-union":
        robin_inlier_indices, semantic_inlier_indices = robin_find_all_maximum_hypercliques(hypergraph, g_nx,
                                                                                            tgt_semantic_id_map)
        return semantic_inlier_indices, robin_inlier_indices, hypergraph, g_nx


def robin_prune_outliers_triplet_order_invariant(tgt, tgt_semantic_id_map, order_db, method='maxclique',
                                                 chirality='left-handed'):
    """ Deprecated """
    assert (tgt.shape[0] == 2)
    assert (tgt.shape[1] == len(tgt_semantic_id_map))
    num_keypoints = tgt.shape[1]
    all_triplets = combinations(list(range(num_keypoints)), 3)

    # creating a Graph in robin
    g = robin_py.AdjListGraph()
    # also create a graph in networkx
    g_nx = nx.Graph()
    for i in range(num_keypoints):
        g.AddVertex(i)
        g_nx.add_node(i)

    # build the compatibility graph
    for triplet in all_triplets:
        # convert triplet indices to semantic ids
        triplet_semantic_ids = (
            tgt_semantic_id_map[triplet[0]], tgt_semantic_id_map[triplet[1]], tgt_semantic_id_map[triplet[2]])
        triplet_points = np.vstack((tgt[:, triplet[0]], tgt[:, triplet[1]], tgt[:, triplet[2]])).T
        consistent = test_triplet_winding_order(triplet_points, triplet_semantic_ids, order_db,
                                                chirality=chirality)
        if consistent:
            # add edges to robin graph
            g.AddEdge(triplet[0], triplet[1])
            g.AddEdge(triplet[0], triplet[2])
            g.AddEdge(triplet[1], triplet[2])
            # add edges to nx graph
            g_nx.add_edge(triplet[0], triplet[1])
            g_nx.add_edge(triplet[0], triplet[2])
            g_nx.add_edge(triplet[1], triplet[2])

    # run robin
    if method == "maxclique":
        inlier_indices = robin_py.FindInlierStructure(
            g, robin_py.InlierGraphStructure.MAX_CLIQUE)
    elif method == "maxcore":
        inlier_indices = robin_py.FindInlierStructure(
            g, robin_py.InlierGraphStructure.MAX_CORE)
    else:
        raise RuntimeError('Prune outliers only support maxclique and maxcore')
    inlier_indices.sort()
    return [tgt_semantic_id_map[i] for i in inlier_indices], inlier_indices, g_nx


def robin_prune_outliers(tgt, cad_dist_min, cad_dist_max, noise_bound, method='maxclique'):
    '''
    First form a compatibility graph and then 
    Use robin to select inliers
    '''
    N = tgt.shape[1]
    si, sj = np.meshgrid(np.arange(N), np.arange(N))
    mask_uppertri = (sj > si)
    si = si[mask_uppertri]
    sj = sj[mask_uppertri]

    # distances || tgt_j - tgt_i ||
    tgt_dist_ij = np.linalg.norm(
        tgt[:, sj] - tgt[:, si], axis=0)  # shape (n-1)_tri

    allEdges = np.arange(si.shape[0])
    check1 = tgt_dist_ij >= (cad_dist_min - 2 * noise_bound)
    check2 = tgt_dist_ij <= (cad_dist_max + 2 * noise_bound)
    mask_compatible = check1 & check2
    validEdges = allEdges[mask_compatible]
    sdata = np.zeros_like(si)
    sdata[mask_compatible] = 1

    comp_mat = np.zeros((N, N))
    comp_mat[si, sj] = sdata

    # creating a Graph in robin
    g = robin_py.AdjListGraph()
    for i in range(N):
        g.AddVertex(i)

    for edge_idx in validEdges:
        # print(f'Add edge between {si[edge_idx]} and {sj[edge_idx]}.')
        g.AddEdge(si[edge_idx], sj[edge_idx])

    if method == "maxclique":
        inlier_indices = robin_py.FindInlierStructure(
            g, robin_py.InlierGraphStructure.MAX_CLIQUE)
    elif method == "maxcore":
        inlier_indices = robin_py.FindInlierStructure(
            g, robin_py.InlierGraphStructure.MAX_CORE)
    else:
        raise RuntimeError('Prune outliers only support maxclique and maxcore')

    # adj_mat = g.GetAdjMat()

    return inlier_indices, comp_mat


def solve_wahba(A, B):
    '''
    solve the Wahba problem using SVD
    '''
    M = B @ A.T
    R = project_to_SO3(M)
    return R


def solve_3dcat_with_altern(tgt, cad_kpts, lam=0.0, weights=None, enforce_csum=True, print_info=False,
                            normalize_lam=False):
    '''
    Solver weighted outlier-free category registration using alternating minimization
    '''
    N = tgt.shape[1]
    K = cad_kpts.shape[0]
    if weights is None:
        weights = np.ones(N)

    if normalize_lam:
        lam_old = lam
        lam = lam_old * float(N) / float(K)
        print(f'Altern: normalize LAM: {lam_old} --> {lam}.')

    if K > 3 * N and lam == 0.0:
        raise RuntimeError('If K is larger than 3N, then lambda has to be strictly positive.')

    wsum = np.sum(weights)
    wsqrt = np.sqrt(weights)

    # compute weighted centers of tgt and cad_kpts
    # tgt has shape 3 by N
    # cad_kpts has shape K by 3 by N
    y_w = tgt @ weights / wsum  # y_w shape (3,)
    b_w = np.sum(
        cad_kpts * weights[np.newaxis, np.newaxis, :], axis=2) / wsum  # b_w shape (K,3)
    # compute relative positions
    Ybar = (tgt - y_w[:, np.newaxis]) * wsqrt[np.newaxis, :]
    ybar = np.reshape(Ybar, (3 * N,), order='F')  # vectorize
    B = (cad_kpts - b_w[:, :, np.newaxis]) * wsqrt[np.newaxis, np.newaxis, :]  # B shape (K,3,N)
    Bbar = np.transpose(
        np.reshape(B, (K, 3 * N), order='F'))  # Bbar shape (3N, K)
    IN = np.identity(N)
    IK = np.identity(K)

    if enforce_csum:
        e = np.ones((K, 1))
        H11 = 2 * (Bbar.T @ Bbar + lam * IK)
        H11inv = np.linalg.inv(H11)
        scalar = e.T @ H11inv @ e
        vector = H11inv @ e
        G = H11inv - (vector @ vector.T) / scalar
        g = vector / scalar
    else:
        H = Bbar.T @ Bbar + lam * IK
        Hinv = np.linalg.inv(H)

    # Initialize
    R = np.identity(3)
    c = np.zeros(K)
    prev_cost = np.inf
    epsilon = 1e-12
    MaxIters = 1e3
    itr = 0
    while itr < MaxIters:
        # Fix R, update c
        if enforce_csum:
            c = 2 * (G @ Bbar.T @ np.kron(IN, R.T) @ ybar) + np.squeeze(g)
        else:
            c = Hinv @ Bbar.T @ np.kron(IN, R.T) @ ybar

        # Fix c, update R
        shape_c = np.sum(
            cad_kpts * c[:, np.newaxis, np.newaxis], axis=0)
        R = solve_wahba(shape_c, Ybar)

        # Compute cost and check convergence
        diff = Ybar - R @ shape_c
        cost = np.sum(diff ** 2) + lam * np.sum(c ** 2)
        cost_diff = np.abs(cost - prev_cost)
        if cost_diff < epsilon:
            if print_info:
                print(f'Altern converges in {itr} iterations, cost diff: {cost_diff}, final cost: {cost}.')
            break
        if itr == MaxIters - 1 and cost_diff > epsilon:
            if print_info:
                print(f'Altern does not converge in {MaxIters} iterations, cost diff: {cost_diff}, final cost: {cost}.')

        itr = itr + 1
        prev_cost = cost

    t = y_w - R @ (b_w.T @ c)

    residuals = calc_residuals(R, t, c, tgt, cad_kpts)

    return R, t, c, itr, residuals


def solve_3dcat_with_sdp(tgt, cad_kpts, lam=0.0, weights=None, enforce_csum=True, print_info=True, normalize_lam=False):
    '''
    Solve weighted outlier-free category registration using SDP relaxation
    '''
    P = get_vectorization_permutation()

    N = tgt.shape[1]
    K = cad_kpts.shape[0]
    if weights is None:
        weights = np.ones(N)

    if normalize_lam:
        lam_old = lam
        lam = lam_old * float(N) / float(K)
        print(f'SDP: normalize LAM: {lam_old} --> {lam}.')

    if K > 3 * N and lam == 0.0:
        raise RuntimeError('If K is larger than 3N, then lambda has to be strictly positive.')

    wsum = np.sum(weights)
    wsqrt = np.sqrt(weights)

    # compute weighted centers of tgt and cad_kpts
    # tgt has shape 3 by N
    # cad_kpts has shape K by 3 by N
    y_w = tgt @ weights / wsum  # y_w shape (3,)
    b_w = np.sum(
        cad_kpts * weights[np.newaxis, np.newaxis, :], axis=2) / wsum  # b_w shape (K,3)
    # compute relative positions
    Ybar = (tgt - y_w[:, np.newaxis]) * wsqrt[np.newaxis, :]
    ybar = np.reshape(Ybar, (3 * N,), order='F')  # vectorize
    B = (cad_kpts - b_w[:, :, np.newaxis]) * wsqrt[np.newaxis, np.newaxis, :]  # B shape (K,3,N)
    Bbar = np.transpose(
        np.reshape(B, (K, 3 * N), order='F'))  # Bbar shape (3N, K)

    IK = np.identity(K)
    I3N = np.identity(3 * N)
    I3 = np.identity(3)
    if enforce_csum:
        # compute H, Hinv and G, g
        e = np.ones((K, 1))
        H11 = 2 * (Bbar.T @ Bbar + lam * IK)
        H11inv = np.linalg.inv(H11)
        scalar = e.T @ H11inv @ e
        vector = H11inv @ e
        G = H11inv - (vector @ vector.T) / scalar
        g = vector / scalar
        # compute M and h
        M = np.concatenate((
            2 * Bbar @ G @ Bbar.T - I3N,
            2 * np.sqrt(lam) * G @ Bbar.T), axis=0)
        h = np.concatenate((
            Bbar @ g, np.sqrt(lam) * g), axis=0)  # 2021 Oct-11: added missing sqrt(lam) before g
    else:
        Hinv = np.linalg.inv(Bbar.T @ Bbar + lam * IK)
        M = np.concatenate((
            Bbar @ Hinv @ Bbar.T - I3N, np.sqrt(lam) * Hinv * Bbar.T), axis=0)
        h = np.zeros((3 * N + K, 1))  # 2021 Oct-11: added missing lam * c'*c

    YkI3 = np.kron(Ybar.T, I3)
    Q = np.block([
        [h.T @ h, h.T @ M @ YkI3 @ P],
        [P.T @ YkI3.T @ M.T @ h, P.T @ YkI3.T @ M.T @ M @ YkI3 @ P]])

    A, b = get_rotation_relaxation_constraints()
    m = len(A)

    # mdic = {
    #     'A': A,
    #     'b': b,
    #     'C': Q
    # }
    # scipy.io.savemat('sample.mat',mdic)

    # Use CVXPY to solve a standard linear SDP
    n = 10
    X = cp.Variable((n, n), symmetric=True)
    constraints = [X >> 0]
    constraints += [
        cp.trace(A[i] @ X) == b[i] for i in range(m)]
    prob = cp.Problem(
        cp.Minimize(cp.trace(Q @ X)), constraints)
    prob.solve(solver=cp.MOSEK)

    Xval = X.value

    if Xval is None:
        print(f'LAM: {lam}, Weights:')
        print(weights)
        raise ArithmeticError("SDP solver failed.")

    # Round solution and check tightness of relaxation
    f_sdp = np.trace(Q @ Xval)

    ev, evec = np.linalg.eig(Xval)
    idx = np.argsort(ev)
    evmax = ev[idx[-1]]
    evsmax = ev[idx[-2]]
    vec = evec[:, idx[-1]]
    vec = vec / vec[0]
    r = vec[1:]
    R = project_to_SO3(
        np.reshape(r, (3, 3), order='F'))
    r = np.reshape(R, (9, 1), order='F')
    rhomo = np.concatenate((
        np.array([[1.0]]), r), axis=0)
    f_est = np.squeeze(
        rhomo.T @ Q @ rhomo)

    rel_gap = abs(f_est - f_sdp) / f_est
    if print_info:
        print(f'SDP relax: lam_1={evmax},lam_2={evsmax},f_sdp={f_sdp},f_est={f_est},rel_gap={rel_gap},lam={lam}.')

    # Recover optimal c and t from R
    IN = np.identity(N)
    if enforce_csum:
        c = 2 * (G @ Bbar.T @ np.kron(IN, R.T) @ ybar) + np.squeeze(g)
    else:
        c = Hinv @ Bbar.T @ np.kron(IN, R.T) @ ybar

    t = y_w - R @ (b_w.T @ c)

    residuals = calc_residuals(R, t, c, tgt, cad_kpts)

    return R, t, c, rel_gap, residuals


def calc_residuals(R, t, c, tgt, cad_kpts):
    '''
    Calculate non-squared residuals 
    '''
    shape_est = np.sum(
        cad_kpts * c[:, np.newaxis, np.newaxis], axis=0)
    shape_transform = R @ shape_est + t[:, np.newaxis]
    residuals = np.sqrt(
        np.sum(
            (tgt - shape_transform) ** 2, axis=0))

    return residuals


def solve_3dcat_ransac(tgt,
                       cad_db,
                       noise_bound,
                       lam=0.0,
                       enforce_csum=True,
                       normalize_lam=False,
                       solver_type='sdp',
                       sample_size=10,
                       random_seed=42,
                       max_iters=5000,
                       min_inlier_ratio_threshold=0.1,
                       save_trajectory=False):
    """ Use RANSAC w/ solvers to solve robust category-level problems
    """

    # repeatable ransac
    ransac_rng = np.random.default_rng(random_seed)

    def ideal_ransac_iter(desired_probability, outlier_probability, sample_size):
        n = np.ceil(np.log(1 - desired_probability) / np.log(1 - (1 - outlier_probability) ** sample_size + 1e-10))
        if n < 0:
            n = np.inf
        return n

    def residual_fun(X, meas_pts, cad_db):
        """ Calculate reprojected residuals """
        R_est = X['estimate'][0]
        t_est = X['estimate'][1].flatten()
        c_est = X['estimate'][2].flatten()
        model_est = np.sum(cad_db * c_est[:, np.newaxis, np.newaxis], axis=0)
        transformed_model_est = R_est @ model_est + t_est[:, np.newaxis]
        residuals = np.linalg.norm(transformed_model_est - meas_pts, axis=0)
        return residuals

    start_time = time.time()
    best_result = None
    best_inlier_ratio = 0
    best_inliers = []
    adapt_max_iters = max_iters
    total_iters = 0
    N = tgt.shape[1]
    for iter in range(max_iters):
        sampled_indices = ransac_rng.choice(N, size=sample_size, replace=False)

        # run solver
        result = solve_3dcat(tgt=tgt[:, sampled_indices],
                             cad_db=cad_db[:, :, sampled_indices],
                             noise_bound=noise_bound,
                             gnc=False,
                             lam=lam,
                             normalize_lam=normalize_lam,
                             enforce_csum=enforce_csum,
                             solver_type=solver_type,
                             save_trajectory=save_trajectory,
                             print_info=False)
        if best_result is None:
            best_result = result

        # build consensus set
        c_residuals = residual_fun(result, tgt, cad_db)
        inlier_indices = c_residuals < noise_bound
        num_inliers = np.count_nonzero(inlier_indices)
        inlier_ratio = num_inliers / N

        #if inlier_ratio > best_inlier_ratio and inlier_ratio >= min_inlier_ratio_threshold and num_inliers >= 3:
        #    # rerun on new inliers
        #    sampled_tgt = tgt[:, inlier_indices]
        #    sampled_tgt_cad_db_array = cad_db[:, :, inlier_indices]
        #    better_result = solve_3dcat(tgt=sampled_tgt,
        #                                cad_db=sampled_tgt_cad_db_array,
        #                                noise_bound=noise_bound,
        #                                gnc=False,
        #                                lam=lam,
        #                                normalize_lam=normalize_lam,
        #                                enforce_csum=enforce_csum,
        #                                solver_type=solver_type,
        #                                save_trajectory=save_trajectory,
        #                                print_info=False)
        #    maybe_better_cost = np.mean(residual_fun(better_result, sampled_tgt, sampled_tgt_cad_db_array))

        #    if maybe_better_cost < c_cost:
        #        result = better_result
        #        c_residuals = residual_fun(better_result, tgt, cad_db)
        #        inlier_indices = c_residuals < noise_bound
        #        num_inliers = np.count_nonzero(inlier_indices)
        #        inlier_ratio = num_inliers / N

        # update inlier ratio
        if inlier_ratio > best_inlier_ratio:
            best_result = result
            best_inlier_ratio = inlier_ratio
            best_inliers = inlier_indices

        # update max iters
        est_outlier_ratio = 1 - inlier_ratio
        adapt_max_itr = min(ideal_ransac_iter(0.99, est_outlier_ratio, sample_size), adapt_max_iters)
        if iter >= adapt_max_itr - 1:
            total_iters = iter + 1
            break

    # do a final solve
    sampled_tgt = tgt[:, best_inliers]
    sampled_tgt_cad_db_array = cad_db[:, :, best_inliers]
    final_result = solve_3dcat(tgt=sampled_tgt,
                               cad_db=sampled_tgt_cad_db_array,
                               noise_bound=noise_bound,
                               gnc=False,
                               lam=lam,
                               normalize_lam=normalize_lam,
                               enforce_csum=enforce_csum,
                               solver_type=solver_type,
                               save_trajectory=save_trajectory,
                               print_info=False)
    final_residuals = residual_fun(final_result, sampled_tgt, sampled_tgt_cad_db_array)
    final_cost = np.mean(final_residuals)

    end_time = time.time()
    debug_print(f"RANSAC + PACE3D {solver_type} solver time: {end_time - start_time}")

    # calculate the final residuals with the best model
    best_residuals = residual_fun(best_result, tgt, cad_db)
    best_cost = np.mean(best_residuals)

    # return the result payload
    result_payload = {
        # meta data
        'solver_func': f'ransac_pace3d_{solver_type}',

        #
        # Estimated
        #
        # transformation data
        'estimate': final_result['estimate'],

        # timing data
        'solver_time': end_time - start_time,
        'itr': total_iters,

        # cost and residuals
        'final_cost': final_cost,
        'final_residuals': final_residuals,

        # inlier data
        'ransac_inlier_indices': np.argwhere(best_inliers).flatten().tolist(),
        "weights": best_inliers,
    }

    return result_payload


def solve_3dcat(tgt, cad_db, noise_bound, lam=0.0, gnc=True, div_factor=1.4, enforce_csum=True, normalize_lam=False,
                solver_type='sdp', save_trajectory=False, print_info=True):
    """ GNC category level registration (3D-3D)
    :argument tgt Input target points
    :argument cad_db Input library of CAD models
    :argument noise_bound maximum allowed inlier residual (non-squared residual)
    :argument lam regularization factor for shape coefficients, lam >= 0
    :argument gnc flag selects if GNC-TLS is used to be robust to outliers (default True)
    :argument div_factor GNC continuation factor (default 1.4)
    :argument enforce_csum flag to enforce the sum of all coefficients equal to 1

    :return: solution dictionary containing estimated R, t, c, and residuals
    """
    if solver_type not in ["sdp", "altern"]:
        raise ValueError("Unrecognized solver: {}".format(solver_type))
    # number of key points
    N = tgt.shape[1]

    if isinstance(cad_db, list):
        K = len(cad_db)
        assert N == cad_db[0]['kpts'].shape[1]
        # If cad_db is a list, then obtain keypoints from list
        cad_kpts = []
        for i in range(len(cad_db)):
            cad_kpts.append(cad_db[i]['kpts'])
        cad_kpts = np.array(cad_kpts)
    else:  # Otherwise, cad_db is already in kpts format as a np array
        cad_kpts = cad_db
        K = cad_kpts.shape[0]
        assert N == cad_kpts.shape[2]

    # If outlier free
    if not gnc:
        if solver_type == "sdp":
            R, t, c, _, residuals = solve_3dcat_with_sdp(tgt, cad_kpts, lam=lam, weights=None,
                                                         enforce_csum=enforce_csum, normalize_lam=normalize_lam,
                                                         print_info=print_info)
            solution = {
                'type': 'category level registration',
                'method': 'sdp',
                'estimate': (R, t, c),
                'residuals': residuals
            }
        elif solver_type == "altern":
            R, t, c, _, residuals = solve_3dcat_with_altern(tgt, cad_kpts, lam=lam, weights=None,
                                                            enforce_csum=enforce_csum, normalize_lam=normalize_lam,
                                                            print_info=print_info)
            solution = {
                'type': 'category level registration',
                'method': 'sdp',
                'estimate': (R, t, c),
                'residuals': residuals
            }
        else:
            raise NotImplementedError("Unknown solver type.")

        if save_trajectory:
            trajectory = [(R, t, c)]

    # Default use GNC-TLS
    else:
        weights = np.ones(N)
        stop_th = 1e-6
        max_steps = 1e2
        barc2 = 1.0
        itr = 0

        pre_TLS_cost = np.inf
        cost_diff = np.inf

        solver_func = solve_3dcat_with_sdp
        if solver_type == "altern":
            solver_func = solve_3dcat_with_altern

        trajectory = []
        while itr < max_steps and cost_diff > stop_th:
            if np.sum(weights) < 1e-12:
                print('GNC encounters numerical issues, the solution is likely to be wrong.')
                break

            #  fix weights and solve for transformation
            try:
                R, t, c, _, residuals = solver_func(tgt, cad_kpts, lam=lam, weights=weights, enforce_csum=enforce_csum,
                                                    print_info=False, normalize_lam=normalize_lam)
            except ArithmeticError:
                print('GNC solver failed, the solution is likely to be wrong.')
                break

            #  fix transformations and update weights
            residuals = residuals / noise_bound
            residuals = residuals ** 2  # residuals normalized by noise_bound

            TLS_cost = np.inner(weights, residuals)
            cost_diff = np.abs(TLS_cost - pre_TLS_cost)

            if itr < 1:
                max_residual = np.max(residuals)
                mu = max(1 / (5 * max_residual / barc2 - 1), 1e-4)
                # mu = 1e-3
                print(f'GNC first iteration max residual: {max_residual}, set mu={mu}.')

            th1 = (mu + 1) / mu * barc2
            th2 = (mu) / (mu + 1) * barc2

            prev_weights = np.copy(weights)

            for i in range(N):
                if residuals[i] - th1 >= 0:
                    weights[i] = 0
                elif residuals[i] - th2 <= 0:
                    weights[i] = 1.0
                else:
                    weights[i] = np.sqrt(
                        barc2 * mu * (mu + 1) / residuals[i]) - mu
                    assert (weights[i] >= 0 and weights[i] <= 1)

            weights_diff = np.linalg.norm(weights - prev_weights)
            weights_sum = np.sum(weights)

            # print('Residuals unsquared:')
            # print(np.sqrt(residuals))
            # print('Weights updated to:')
            # print(weights)

            # print(f'Itr: {itr}, weights_diff: {weights_diff}, weights_sum: {weights_sum}, cost_diff: {cost_diff}.')

            #  increase mu
            mu = mu * div_factor
            itr = itr + 1
            pre_TLS_cost = TLS_cost

            if save_trajectory:
                trajectory.append((R, t, c))

        solution = {
            'type': 'category level registration',
            'method': 'gnc',
            'estimate': (R, t, c),
            'max_steps': max_steps,
            'weights': weights,
            'itr': itr,
            'div_factor': div_factor
        }

    # calculate final residuals and add to solution dictionary
    residuals = calc_residuals(R, t, c, tgt, cad_kpts)
    normalized_residuals = residuals / noise_bound

    solution['residuals'] = residuals
    solution['normalized_residuals'] = normalized_residuals
    if save_trajectory:
        solution['solver_trajectory'] = trajectory

    return solution


def solve_3dcat_irls(tgt, cad_db, noise_bound, lam=0.0, enforce_csum=True, robust_fun='TLS', normalize_lam=False):
    '''
    Solve robust estimation using iterative reweighted least squares
    '''
    # number of key points
    N = tgt.shape[1]

    if isinstance(cad_db, list):
        K = len(cad_db)
        assert N == cad_db[0]['kpts'].shape[1]
        # Obtain keypoints from cad_db
        cad_kpts = []
        for i in range(len(cad_db)):
            cad_kpts.append(cad_db[i]['kpts'])
        cad_kpts = np.array(cad_kpts)
    else:
        cad_kpts = cad_db
        K = cad_kpts.shape[0]
        assert N == cad_kpts.shape[2]

    weights = np.ones(N)
    stop_th = 1e-6
    max_steps = 1e3
    barc2 = 1.0
    if robust_fun == "GM":
        max_steps = 100
        sigma = 1.0 * barc2

    itr = 0
    prev_cost = np.inf
    cost_diff = np.inf

    while itr < max_steps and cost_diff > stop_th:
        if np.sum(weights) < 1e-12:
            print('IRLS encounters numerical issues, the solution is likely to be wrong.')
            break

        #  fix weights and solve for transformation
        try:
            R, t, c, _, residuals = solve_3dcat_with_sdp(tgt, cad_kpts, lam=lam, weights=weights, enforce_csum=enforce_csum,
                                                         print_info=False, normalize_lam=normalize_lam)
        except ArithmeticError:
            print('IRLS solver failed, the solution is likely to be wrong.')
            break

        #  calculate residuals
        residuals = residuals / noise_bound
        residuals_sq = residuals ** 2

        if robust_fun == "TLS":
            # Check convergence
            cost = np.inner(weights, residuals)
            cost_diff = np.abs(cost - prev_cost)
            # Update weights
            weights = np.zeros(N)
            weights[residuals < barc2] = 1.0
        elif robust_fun == "GM":
            # Check convergence
            cost = np.sum(residuals_sq / (sigma + residuals_sq))
            cost_diff = np.abs(cost - prev_cost)
            # Update weights
            weights = sigma ** 2 / ((sigma + residuals_sq) ** 2)
        else:
            raise RuntimeError('IRLS only supports TLS and GM function now.')

        prev_cost = cost
        itr = itr + 1

    solution = {
        'type': 'category level registration',
        'method': 'irls',
        'robust_fun': robust_fun,
        'estimate': (R, t, c),
        'max_steps': max_steps,
        'weights': weights,
        'itr': itr
    }
    residuals = calc_residuals(R, t, c, tgt, cad_kpts)
    normalized_residuals = residuals / noise_bound

    solution['residuals'] = residuals
    solution['normalized_residuals'] = normalized_residuals
    return solution
