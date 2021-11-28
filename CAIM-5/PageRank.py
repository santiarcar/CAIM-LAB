#!/usr/bin/python

from collections import namedtuple
import time
import sys
import re


# This class stores the basic information of every edge -> flight
class Edge:
    def __init__(self, origin=None, index=None):
        self.origin = origin
        self.weight = 1
        self.index = index

    def __repr__(self):
        return "edge: {0} {1}".format(self.origin, self.weight)


class Airport:
    def __init__(self, iden=None, name=None):
        self.code = iden
        self.name = name
        # routes will store the edges (flights) that arrive to this airport (this airport is their destination)
        self.routes = []
        self.routeHash = dict()
        self.outweight = 0

    def addEdge(self, origin) -> bool:
        if origin in self.routeHash:
            idx = self.routeHash[origin]
            self.routes[idx].weight += 1
            return False
        else:
            new_edge = Edge(origin=origin, index=airportHash[origin].index)
            self.routes.append(new_edge)
            self.routeHash[origin] = len(self.routes) - 1
            return True

    def __repr__(self):
        return f"{self.code}\t{self.name}"


edgeList = []  # list of Edge
edgeHash = dict()  # hash of edge to ease the match
airportList = []  # list of Airport
airportHash = dict()  # hash key IATA code -> Airport


def readAirports(fd):
    print("Reading Airport file from {0}".format(fd))
    airportsTxt = open(fd, "r");
    cont = 0
    for line in airportsTxt.readlines():
        a = Airport()
        try:
            temp = line.split(',')
            if len(temp[4]) != 5:
                raise Exception('not an IATA code')
            a.name = temp[1][1:-1] + ", " + temp[3][1:-1]
            a.code = temp[4][1:-1]
            a.index = cont
        except Exception as inst:
            pass
        else:
            cont += 1
            airportList.append(a)
            airportHash[a.code] = a
    airportsTxt.close()
    print(f"There were {cont} Airports with IATA code")


def getAirport(code):
    if code in airportHash:
        return airportHash[code]
    else:
        raise Exception(f"The airport {code} appears on a route but not on the airport file")


def readRoutes(fd):
    print(f"Reading Routes file from {fd}")
    """
    airline_code
    OF_airline_code
    IATA_origin
    OF_Origin
    IATA_destination
    OF_destination
    noise
    """
    airport_count = 0
    route_count = 0
    routesTxt = open(fd, "r")
    for line in routesTxt.readlines():
        try:
            line_terms = line.split(",")
            if (len(line_terms[2]) != 3) or (not re.search("[a-zA-Z]+", line_terms[2])):
                raise Exception("{0} is not IATA".format(line_terms[2]))
            if (len(line_terms[4]) != 3) or (not re.search("[a-zA-Z]+", line_terms[4])):
                raise Exception("{0} is not IATA".format(line_terms[4]))

            iata_origin = line_terms[2]
            iata_dest = line_terms[4]

            airport_origin = getAirport(iata_origin)
            airport_dest = getAirport(iata_dest)

            airport_origin.outweight += 1
            if airport_dest.addEdge(iata_origin):
                airport_count += 1
            route_count += 1

        except Exception as e:
            pass

    print(f"Correct routes found : {route_count}\nAirports found in routes : {airport_count}")


def computePageRanks():
    n = len(airportList)
    P = [1 / n] * n
    L = 0.85

    stopping_threshold = 1e-12
    one_minus_L_avg = (1 - L) / n
    dead_end_weight = 1 / n
    dead_end_factor = (L / n) * n_dead_ends

    n_iters = 0
    stop = False
    while not stop:
        Q = [0] * n

        for i in range(n):
            a = airportList[i]
            summation = 0
            for e in a.routes:
                w_i_j = e.weight
                n_out_j = airportList[e.index].outweight
                summation += P[e.index] * w_i_j / n_out_j

            Q[i] = L * summation + one_minus_L_avg + dead_end_weight * dead_end_factor

        dead_end_weight = one_minus_L_avg + dead_end_weight * dead_end_factor

        stop = all(
            list(
                map(
                    lambda diff: diff < stopping_threshold,
                    [abs(old_value - new_value) for old_value, new_value in zip(P, Q)]
                )
            )
        )
        # Check sum of P
        # print(f'i={n_iters}\tsum(P)={round(sum(P),5)}')
        n_iters += 1
        P = Q

    global pageRank
    pageRank = P
    return n_iters


def outputPageRanks():
    airport_pr = sorted(zip(airportList, pageRank), key=lambda z: z[1], reverse=True)
    print("""
     ############################################################################### 
     ............... Page Rank --- Airport name (Airport code) ..................... 
     ############################################################################### 
     """)
    
    for a, pr in airport_pr:
        print(f"{round(pr, 10)} --- {a.name} ({a.code})")

def main(argv=None):
    readAirports("airports.txt")
    readRoutes("routes.txt")

    global n_dead_ends
    n_dead_ends = sum(map(lambda a: a.outweight == 0, airportList))

    print(f"There are {n_dead_ends} airports that are dead ends")

    time1 = time.time()
    iterations = computePageRanks()
    time2 = time.time()
    outputPageRanks()
    print("#Iterations:", iterations)
    print("Time of computePageRanks():", time2 - time1)

if __name__ == "__main__":
    sys.exit(main())
