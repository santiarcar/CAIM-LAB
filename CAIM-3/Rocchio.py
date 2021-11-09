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
    s = Search(using=client, index=index)
    q = Q('match', path=path)  # exact search in the path field
    s = s.query(q)
    result = s.execute()

    lfiles = [r for r in result]
    if len(lfiles) == 0:
        raise NameError(f'File [{path}] not found')
    else:
        return lfiles[0].meta.id


def document_term_vector(client, index, id):
    """
    Returns the term vector of a document and its statistics a two sorted list of pairs (word, count)
    The first one is the frequency of the term in the document, the second one is the number of documents
    that contain the term

    :param client:
    :param index:
    :param id:
    :return:
    """
    termvector = client.termvectors(index=index, id=id, fields=['text'],
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

    :param file:
    :return:
    """

    # Get the frequency of the term in the document, and the number of documents
    # that contain the term
    file_tv, file_df = document_term_vector(client, index, file_id)

    max_freq = max([f for _, f in file_tv])

    dcount = doc_count(client, index)
    tfidfw = []
    for (t, w), (_, df) in zip(file_tv, file_df):
        tf_di = w / max_freq
        idf_i = np.log2(dcount / df)
        tfidfw.append((t, tf_di * idf_i))

    return normalize(tfidfw)


def normalize(tw):
    """
    Normalizes the weights in t so that they form a unit-length vector
    It is assumed that not all weights are 0
    :param tw:
    :return:
    """

    mag = 0

    for (_, freq) in tw:
        mag += freq**2
    mag = np.sqrt(mag)

    return [(t, freq/mag) for (t, freq) in tw]


def get_dict_from_query(query: str):
    d = {}
    for word in query:
        word_values = {}
    if '~' in word:
        res_word = word.split('~')[0]
        value = word.split('~')[1]
        d[res_word] = {'sign': '~',
                       'value': value}
    elif '^' in word:
        res_word = word.split('^')[0]
        value = word.split('^')[1]
        d[res_word] = {'sign': '^',
                       'value': value}
    else:
        d[word] = {}

    return d


def get_query_from_dict(d: dict):
    query = []
    for word in d:
        if word.__len__() > 0:
            query.append(f'{word}{d[word["sign"]]}{d[word["value"]]}')
        else:
            query.append(f'{word}')

    return query


def computeTFIDF(documents, client, index):
    tfidf_l = []
    for doc in documents:
        file_id = search_file_by_path(client, index, doc.path)
        tfidf_doc = toTFIDF(client=client,
                            index=index,
                            file_id=file_id,
                            )
        tfidf_l.append(tfidf_doc)
    return tfidf_l


def get_term_relevance(d_query, tfidf_l):
    tfidf_result = dict.fromkeys(d_query.keys(), 0)

    for doc in tfidf_l:
        for word,_  in d_query.items():
            tfidf_result[word] += tfidf_doc[word]

    #print('tfidf result', d_query)
    return tfidf_result


def obtain_most_relevant_docs():
    if query is not None:
        q = Q('query_string', query=query[0])
        for i in range(1, len(query)):
            q &= Q('query_string', query=query[i])
        s = s.query(q)
        response = s[0:k].execute()
    return response

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=True, default=None, help='Path to the files')
    parser.add_argument('--index', required=True, default=None, help='Index for the files')
    parser.add_argument('--k', default=10, type=int, help='Number of top documents considered relevant for each round')
    parser.add_argument('--nrounds', default=10, type=int, help='Number of applications of Roccchio\'s rule')
    parser.add_argument('--R', default=10, type=int, help='Maximum number of terms to be ke in the new query')
    parser.add_argument('--alpha', default=10, type=int, help='Value fro "alpha", first weight of the Rochio rule')
    parser.add_argument('--beta', default=10, type=int, help='Value fro "beta", second weight of the Rochio rule')
    parser.add_argument('--query', default=None, nargs=argparse.REMAINDER, help='List of words to search')

    args = parser.parse_args()

    path = args.path
    index = args.index
    k = args.k
    nrounds = args.nrounds
    R = args.R
    alpha = args.alpha
    beta = args.beta
    query = args.query

    client = Elasticsearch()
    s = Search(using=client, index=index)

    for i in range(nrounds):
        # Obtain most relevant docs
        # Apply Rocchio

    # Return k most relevant



















    # check if the filters are valid
    for f in args.filter:
        if f not in ['lowercase', 'asciifolding', 'stop', 'stemmer', 'porter_stem', 'kstem', 'snowball']:
            raise NameError(
                'Invalid filter must be a subset of: lowercase, asciifolding, stop, porter_stem, kstem, snowball')

    ldocs = []

    # Reads all the documents in a directory tree and generates an index operation for each
    lfiles = generate_files_list(path)
    print('Indexing %d files' % len(lfiles))
    print('Reading files ...')
    for f in lfiles:
        ftxt = codecs.open(f, "r", encoding='iso-8859-1')
        text = ''
        for line in ftxt:
            text += line
        # Insert operation for a document with fields' path' and 'text'
        ldocs.append({'_op_type': 'index', '_index': index, 'path': f, 'text': text})

    client = Elasticsearch(timeout=1000)

    # Tokenizers: whitespace classic standard letter
    my_analyzer = analyzer('default',
                           type='custom',
                           tokenizer=tokenizer(args.token),
                           filter=args.filter
                           )

    try:
        # Drop index if it exists
        ind = Index(index, using=client)
        ind.delete()
    except NotFoundError:
        pass
    # then create it
    ind.settings(number_of_shards=1)
    ind.create()
    ind = Index(index, using=client)

    # configure default analyzer
    ind.close()  # index must be closed for configuring analyzer
    ind.analyzer(my_analyzer)

    # configure the path field so it is not tokenized and we can do exact match search
    client.indices.put_mapping(doc_type='document', index=index, include_type_name=True, body={
        "properties": {
            "path": {
                "type": "keyword",
            }
        }
    })

    ind.save()
    ind.open()
    print("Index settings=", ind.get_settings())
    # Bulk execution of elastic search operations (faster than executing all one by one)
    print('Indexing ...')
    bulk(client, ldocs)
