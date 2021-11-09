from elasticsearch import Elasticsearch
from elasticsearch.exceptions import NotFoundError

import argparse

from elasticsearch_dsl import Search
from elasticsearch_dsl.query import Q
from elasticsearch.client import CatClient

import numpy as np
import operator

"""
Functions provided by the teachers in previous sessions
"""


def doc_count(client, index):
    return int(CatClient(client).count(index=[index], format='json')[0]['count'])


def search_file_by_path(client, index, path):
    q = Q('match', path=path)  # exact search in the path field
    s = Search(using=client, index=index)

    res = s.query(q)
    result = res.execute()

    lfiles = [r for r in result]
    if len(lfiles) == 0:
        raise NameError(f'File [{path}] not found')
    else:
        return lfiles[0].meta.id


def document_term_vector(client, index, file_id):
    """
    Returns the term vector of a document and its statistics a two sorted list of pairs (word, count)
    The first one is the frequency of the term in the document, the second one is the number of documents
    that contain the term

    :return:
    """
    termvector = client.termvectors(index=index, id=file_id, fields=['text'],
                                    positions=False, term_statistics=True)

    file_td = {}
    file_df = {}

    if 'text' in termvector['term_vectors']:
        for t in termvector['term_vectors']['text']['terms']:
            file_td[t] = termvector['term_vectors']['text']['terms'][t]['term_freq']
            file_df[t] = termvector['term_vectors']['text']['terms'][t]['doc_freq']
    return sorted(file_td.items()), sorted(file_df.items())


"""
Functions completed in previous sessions
"""


def toTFIDF(client, index, file_id):
    """
    Returns the term weights of a document

    :param file_id:
    :return:
    """

    # Get the frequency of the term in the document, and the number of documents
    # that contain the term
    file_tv, file_df = document_term_vector(client=client,
                                            index=index,
                                            file_id=file_id)

    max_freq = max([f for _, f in file_tv])

    dcount = doc_count(client, index)
    tfidfw = {}
    for (t, w), (_, df) in zip(file_tv, file_df):
        tf_di = w / max_freq
        idf_i = np.log2(dcount / df)
        tfidfw[t] = tf_di * idf_i

    return normalize(tfidfw)


def normalize(tw):
    """
    Normalizes the weights in t so that they form a unit-length vector
    It is assumed that not all weights are 0
    :param tw:
    :return:
    """

    mag = 0

    for key in tw:
        mag += tw[key] ** 2
    mag = np.sqrt(mag)

    return {key: tw[key] / mag for key in tw}


def get_dict_from_query(query):
    d = {}
    for term in query:
        if '^' in term:
            main_term = term.split('^')[0]
            value = term.split('^')[1]
        else:
            main_term = term
            value = 1

        if '~' in main_term:
            word = main_term.split('~')[0]
        else:
            word = main_term

        d[term] = {'word': word,
                   'importance': float(value),
                   }

    return d


def get_query_from_dict(d: dict):
    q = []
    for term in d:
        print(f'TERM: {term}\n{term.split("^")[0]}\n{d[term]["importance"]}')
        q.append(f'{term.split("^")[0]}^{d[term]["importance"]}')

    return q


def computeTFIDF(client, index, documents):
    tfidf_l = []
    for doc in documents:
        file_id = search_file_by_path(client=client,
                                      index=index,
                                      path=doc.path)
        tfidf_doc = toTFIDF(client=client,
                            index=index,
                            file_id=file_id)
        tfidf_l.append(tfidf_doc)
    return tfidf_l


def get_term_relevance(d_query, tfidf_l):
    tfidf_result = dict.fromkeys(d_query.keys(), 0)

    for tfidf_doc in tfidf_l:
        print(f"d_query:\n{d_query}")
        for term, information in d_query.items():
            tfidf_result[term] += (tfidf_doc.get(information['word']) or 0)

    return tfidf_result


def get_most_relevant_docs(query):
    if query is not None:
        q = Q('query_string', query=query[0])
        s = Search(using=client, index=index)
        for i in range(1, len(query)):
            q &= Q('query_string', query=query[i])
        response = s.query(q)
        return response[0:k].execute()


def reorder_query(query):
    for j in range(len(query)):
        if ('^' in query[j]) and ('~' in query[j]):
            boost_pos = query[j].find('^')
            fuzzy_pos = query[j].find('~')
            if boost_pos < fuzzy_pos:
                w = query[j].split('^')[0]
                w_tail = query[j].split('^')[1]
                boost = w_tail.split('~')[0]
                fuzzy = w_tail.split('~')[1]
                query[j] = f'{w}~{fuzzy}^{boost}'
    return query


def Rocchio(alpha, beta, dict_query, tfidf_dict):
    new_dict_query = {}
    for term, information in dict_query.items():
        print(f"len(tfidf_dict) : {len(tfidf_dict)}")
        mean_weight = tfidf_dict[term] / len(tfidf_dict)
        new_query_importance = float(alpha) * information['importance'] + float(beta) * mean_weight
        new_dict_query[term] = {'word': information['word'],
                                'importance': new_query_importance,
                                }
    return new_dict_query


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', required=True, default=None, help='Index for the files')
    parser.add_argument('--k', default=10, type=int, help='Number of top documents considered relevant for each round')
    parser.add_argument('--nrounds', default=10, type=int, help='Number of applications of Roccchio\'s rule')
    parser.add_argument('--R', default=10, type=int, help='Maximum number of terms to be ke in the new query')
    parser.add_argument('--alpha', default=1, type=int, help='Value fro "alpha", first weight of the Rochio rule')
    parser.add_argument('--beta', default=1.5, type=int, help='Value fro "beta", second weight of the Rochio rule')
    parser.add_argument('--query', required=True, default=None, nargs=argparse.REMAINDER,
                        help='List of words to search')

    args = parser.parse_args()

    index = args.index
    k = args.k
    nrounds = args.nrounds
    R = args.R
    alpha = args.alpha
    beta = args.beta
    query = args.query

    # For simplicity, it's better to have the boost operator at the end of every term
    query = reorder_query(query)

    client = Elasticsearch()

    # tfidf_l = computeTFIDF(rel_docs)

    for i in range(nrounds):
        # Obtain most relevant docs
        if i > 0:
            print(query)
        rel_docs = get_most_relevant_docs(query)

        tfidf_l = computeTFIDF(client=client,
                               index=index,
                               documents=rel_docs)
        # Apply Rocchio
        dict_query = get_dict_from_query(query=query)
        tfidf_query = get_term_relevance(d_query=dict_query,
                                         tfidf_l=tfidf_l)

        dict_new_query = Rocchio(alpha=alpha,
                                 beta=beta,
                                 dict_query=dict_query,
                                 tfidf_dict=tfidf_query)

        query = get_query_from_dict(dict_new_query)

    rel_docs = get_most_relevant_docs(query)

    for r in rel_docs:  # only returns a specific number of results
        print(f'ID= {r.meta.id} SCORE={r.meta.score}')
        print(f'PATH= {r.path}')
        print(f'TEXT: {r.text[:50]}')
        print('-----------------------------------------------------------------')

