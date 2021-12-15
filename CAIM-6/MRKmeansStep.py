"""
.. module:: MRKmeansDef

MRKmeansDef
*************

:Description: MRKmeansDef



:Authors: bejar


:Version:

:Created on: 17/07/2017 7:42

"""

from mrjob.job import MRJob
from mrjob.step import MRStep

__author__ = 'bejar'


class MRKmeansStep(MRJob):
    prototypes = {}

    def jaccard(self, prot, doc):
        """
        Compute here the Jaccard similarity between  a prototype and a document
        prot should be a list of pairs (word, probability)
        doc should be a list of words
        Words must be alphabeticaly ordered

        The result should be always a value in the range [0,1]
        """

        intersection_size = iter_prot = iter_doc = 0

        while (iter_prot < len(prot)) and (iter_doc < len(doc)):
            if prot[iter_prot][0] == doc[iter_doc]:
                intersection_size += prot[iter_prot][1]
                iter_prot += 1
                iter_doc += 1
            elif prot[iter_prot][0] < doc[iter_doc]:
                iter_prot += 1
            else:
                iter_doc += 1

        norm_squared_proto = 0
        for i in range(len(prot)):
            norm_squared_proto += prot[i][1]**2

        norm_squared_doc = len(doc)

        return intersection_size / (norm_squared_proto + norm_squared_doc - intersection_size)

    def configure_args(self):
        """
        Additional configuration flag to get the prototypes files

        :return:
        """
        super(MRKmeansStep, self).configure_args()
        self.add_file_arg('--prot')

    def load_data(self):
        """
        Loads the current cluster prototypes

        :return:
        """
        f = open(self.options.prot, 'r')
        for line in f:
            cluster, words = line.split(':')
            cp = []
            for word in words.split():
                cp.append((word.split('+')[0], float(word.split('+')[1])))
            self.prototypes[cluster] = cp

    def assign_prototype(self, _, line):
        """
        This is the mapper it should compute the closest prototype to a document

        Words should be sorted alphabetically in the prototypes and the documents

        This function has to return at list of pairs (prototype_id, document words)

        You can add also more elements to the value element, for example the document_id
        """

        # Each line is a string docid:wor1 word2 ... wordn
        doc, words = line.split(':')
        lwords = words.split()

        #
        # Compute map here
        #

        min_dist = float('inf')
        closest_proto = None
        for key in self.prototypes:
            dist_to_proto = 1 - self.jaccard(self.prototypes[key], lwords)
            if(dist_to_proto < min_dist):
                min_dist = dist_to_proto
                closest_proto = key

        # Return pair key, value
        yield closest_proto, (doc, lwords)

    def aggregate_prototype(self, key, values):
        """
        input is cluster and all the documents it has assigned
        Outputs should be at least a pair (cluster, new prototype)

        It should receive a list with all the words of the documents assigned for a cluster

        The value for each word has to be the frequency of the word divided by the number
        of documents assigned to the cluster

        Words are ordered alphabetically but you will have to use an efficient structure to
        compute the frequency of each word

        :param key:
        :param values:
        :return:
        """

        new_proto = {}
        new_proto_docs = []
        docs_in_cluster = 0
        for doc in values:
            docs_in_cluster += 1
            new_proto_docs.append(doc[0])
            for word in doc[1]:
                if word in new_proto:
                    new_proto[word] += 1
                else:
                    new_proto[word] = 1

        return_proto = []
        for word in new_proto:
            return_proto.append((word, new_proto[word] / docs_in_cluster))

        yield key, (sorted(new_proto_docs), sorted(return_proto, key=lambda x: x[0]))

    def steps(self):
        return [MRStep(mapper_init=self.load_data, mapper=self.assign_prototype,
                       reducer=self.aggregate_prototype)
            ]


if __name__ == '__main__':
    MRKmeansStep.run()