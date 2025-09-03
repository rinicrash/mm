#reading from file
# Reading a text file
def read_text_file(file_path):
    # Method 1: Read entire file as a single string
    print("Reading entire file:")
    with open(file_path, 'r') as file:
        content = file.read()
        print(content)
    
    # Method 2: Read line by line
    print("\nReading line by line:")
    with open(file_path, 'r') as file:
        for line in file:
            print(line.strip())  # strip() removes leading/trailing whitespace
    
    # Method 3: Read all lines into a list
    print("\nReading all lines into a list:")
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            print(line.strip())

# Example usage
try:
    read_text_file('example.txt')
except FileNotFoundError:
    print("Error: The file 'example.txt' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")



import csv

# Reading a CSV file
def read_csv_file(file_path):
    # Method 1: Read CSV as a list of rows
    print("Reading CSV as rows:")
    with open(file_path, 'r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            print(row)  # Each row is a list of values
    
    # Method 2: Read CSV with headers using DictReader
    print("\nReading CSV with headers:")
    with open(file_path, 'r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            print(row)  # Each row is a dictionary with column headers as keys

# Example usage
try:
    read_csv_file('example.csv')
except FileNotFoundError:
    print("Error: The file 'example.csv' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")

###########################################################################################################


#my_boolean
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 19:12:48 2025

@author: shwet
"""

documents = [
    {"doc_id": "D1", "title": "Information requirement", "content": "query considers the user feedback as information requirement to search"},
    {"doc_id": "D2", "title": "Information retrieval", "content": "query depends on the model of information retrieval used"},
    {"doc_id": "D3", "title": "Prediction problem", "content": "Many problems in information retrieval can be viewed as prediction problems"},
    {"doc_id": "D4", "title": "Search", "content": "A search engine is one of applications of information retrieval models"},
    {"doc_id": "D5", "title": "Feedback", "content": "feedback is typically used by the system to modify the query and improve prediction"},
    {"doc_id": "D6", "title": "information retrieval", "content": "ranking in information retrieval algorithms depends on user query"}
]

import nltk
from nltk.stem import PorterStemmer
import pandas as pd
from collections import defaultdict,Counter

ps=PorterStemmer()

def preprocess(text):
    words=text.lower().split()
    tokens=[ps.stem(word,to_lowercase=True) for word in words]
    return tokens
    
def stopwords(docs):
    token=[]
    for doc in docs:
        token.extend(preprocess(doc["content"]))
    freq=Counter(token).most_common(10)
    print(freq)
    return freq
    
def InvertedIndex(docs,stopwords):
    inverted_index=defaultdict(list)
    for doc in docs:
         tokens=preprocess(doc["content"])
         tokens=[token for token in tokens if token not in stopwords]
         for token in set(tokens):
             inverted_index[token].append(doc["doc_id"])
    return inverted_index

def EvaluateBooleanQuery(invertedIndex,query,docs):
    tokens=query.lower().split()
    tokens=[ps.stem(token) for token in tokens]
    all_docs=set([doc["doc_id"] for doc in docs])
    operator=None
    result=None
    i=0
    while i<len(tokens):
        token=tokens[i]
        if token in ("and", "not", "or"):
            operator = token
            i+=1
            continue
        current_docs=set(invertedIndex.get(token,[]))
        if operator == "not":
            result=all_docs-current_docs
            operator = None
        
        if result is None:
            result=current_docs
        elif operator=="and":
            result=result.intersection(current_docs)
            operator=None
        elif operator=="or":
            result=result.union(current_docs)
            operator=None
        i+=1
    return result if result is not None else set()

#print(preprocess(documents[0]["content"]))
#stopwords(documents)

sw=stopwords(documents)
index=InvertedIndex(documents, sw)
print(EvaluateBooleanQuery(index, "retrieval OR prediction", documents))

###########################################################################################
#inverted index boolean
class InvertedIndex:
    def __init__(self, docs):
        self.docs = docs
        self.index = {}
        self._build()
    
    def _build(self):
        """Build inverted index"""
        for i, doc in enumerate(self.docs):
            for term in set(doc.lower().split()):
                if term not in self.index:
                    self.index[term] = []
                self.index[term].append(i)
    
    def get(self, term):
        """Get posting list"""
        return self.index.get(term.lower(), [])
    
    def AND(self, list1, list2):
        """Intersect two sorted lists"""
        return [x for x in list1 if x in list2]
    
    def OR(self, list1, list2):
        """Union two lists"""
        return sorted(set(list1 + list2))
    
    def NOT(self, posting_list):
        """Complement of posting list"""
        all_docs = list(range(len(self.docs)))
        return [x for x in all_docs if x not in posting_list]
    
    def optimize_terms(self, terms, operation='and'):
        """Sort terms by posting list length for optimal processing"""
        term_lengths = [(term, len(self.get(term))) for term in terms]
        
        if operation == 'and':
            # For AND: process shortest lists first (fewer intersections)
            return [term for term, _ in sorted(term_lengths, key=lambda x: x[1])]
        else:  # OR
            # For OR: process longest lists first (build result faster)
            return [term for term, _ in sorted(term_lengths, key=lambda x: x[1], reverse=True)]
    
    def search(self, query):
        """Optimized boolean search"""
        q = query.lower()
        
        if ' and ' in q:
            terms = [t.strip() for t in q.split(' and ')]
            # Optimize: shortest posting lists first
            terms = self.optimize_terms(terms, 'and')
            result = self.get(terms[0])
            for term in terms[1:]:
                result = self.AND(result, self.get(term))
                if not result:  # Early termination
                    break
            return result
        
        elif ' or ' in q:
            terms = [t.strip() for t in q.split(' or ')]
            # Optimize: longest posting lists first
            terms = self.optimize_terms(terms, 'or')
            result = self.get(terms[0])
            for term in terms[1:]:
                result = self.OR(result, self.get(term))
            return result
        
        elif ' not ' in q:
            pos, neg = q.split(' not ')
            pos_list = self.get(pos.strip())
            neg_list = self.get(neg.strip())
            return self.AND(pos_list, self.NOT(neg_list))
        
        else:
            return self.get(q)

# Usage with optimization demo
docs = ["cat dog bird", "dog bird", "cat mouse", "bird eagle", "mouse cat"]
idx = InvertedIndex(docs)
print("Index:", idx.index)

# Show optimization in action
print("\nQuery: 'cat and bird and dog'")
terms = ['cat', 'bird', 'dog']
print("Posting list sizes:")
for term in terms:
    print(f"  {term}: {len(idx.get(term))} docs")

optimized = idx.optimize_terms(terms, 'and')
print(f"Optimized order: {optimized}")  # Shortest first
print(f"Result: {idx.search('cat and bird and dog')}")

print(f"\nOR optimization:")
or_optimized = idx.optimize_terms(terms, 'or') 
print(f"OR order: {or_optimized}")  # Longest first


#333333333333333######################################################################

#term doc boolean
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 31 11:53:28 2025

@author: preth
"""

class TermDocumentBooleanModel:
    def __init__(self, documents):
        self.documents = documents
        self.vocab = []
        self.term_doc_matrix = []
        self._build_matrix()
    
    def _build_matrix(self):
        """Build term-document matrix"""
        # Get all unique terms
        all_terms = set()
        processed_docs = []
        
        for doc in self.documents:
            tokens = doc.lower().split()
            processed_docs.append(tokens)
            all_terms.update(tokens)
        
        self.vocab = sorted(all_terms)
        
        # Build matrix: rows=terms, cols=documents
        self.term_doc_matrix = []
        for term in self.vocab:
            row = []
            for doc_tokens in processed_docs:
                row.append(1 if term in doc_tokens else 0)
            self.term_doc_matrix.append(row)
    
    def get_term_vector(self, term):
        """Get document vector for a term"""
        term = term.lower()
        if term not in self.vocab:
            return [0] * len(self.documents)
        
        term_idx = self.vocab.index(term)
        return self.term_doc_matrix[term_idx]
    
    def boolean_and(self, term1, term2):
        """Boolean AND operation"""
        vec1 = self.get_term_vector(term1)
        vec2 = self.get_term_vector(term2)
        return [a & b for a, b in zip(vec1, vec2)]
    
    def boolean_or(self, term1, term2):
        """Boolean OR operation"""
        vec1 = self.get_term_vector(term1)
        vec2 = self.get_term_vector(term2)
        return [a | b for a, b in zip(vec1, vec2)]
    
    def boolean_not(self, term):
        """Boolean NOT operation"""
        vec = self.get_term_vector(term)
        return [1 - x for x in vec]
    
    def search(self, query):
        """Search with boolean operators"""
        query = query.lower().strip()
        
        # Single term
        if ' ' not in query:
            result_vector = self.get_term_vector(query)
        
        # AND operation
        elif ' and ' in query:
            terms = [t.strip() for t in query.split(' and ')]
            result_vector = self.get_term_vector(terms[0])
            for term in terms[1:]:
                term_vec = self.get_term_vector(term)
                result_vector = [a & b for a, b in zip(result_vector, term_vec)]
        
        # OR operation
        elif ' or ' in query:
            terms = [t.strip() for t in query.split(' or ')]
            result_vector = self.get_term_vector(terms[0])
            for term in terms[1:]:
                term_vec = self.get_term_vector(term)
                result_vector = [a | b for a, b in zip(result_vector, term_vec)]
        
        # NOT operation
        elif ' not ' in query:
            parts = query.split(' not ')
            pos_term = parts[0].strip()
            neg_term = parts[1].strip()
            
            pos_vec = self.get_term_vector(pos_term)
            neg_vec = self.get_term_vector(neg_term)
            neg_vec = [1 - x for x in neg_vec]  # NOT operation
            result_vector = [a & b for a, b in zip(pos_vec, neg_vec)]
        
        else:
            result_vector = [0] * len(self.documents)
        
        # Return document IDs where result is 1
        return [i for i, val in enumerate(result_vector) if val == 1]
    
    def print_matrix(self):
        """Print term-document matrix"""
        print("Term-Document Matrix:")
        print("Terms\\Docs", end="")
        for i in range(len(self.documents)):
            print(f"\tD{i}", end="")
        print()
        
        for i, term in enumerate(self.vocab):
            print(f"{term:<10}", end="")
            for val in self.term_doc_matrix[i]:
                print(f"\t{val}", end="")
            print()

# Usage Example
if __name__ == "__main__":
    # Sample documents
    docs = [
        "information retrieval system",
        "database search query",
        "information system database",
        "web search engine",
        "query processing system"
    ]
    
    model = TermDocumentBooleanModel(docs)
    model.print_matrix()
    
    print("\nSearch Results:")
    print("'information':", model.search("information"))
    print("'information and system':", model.search("information and system"))
    print("'search or query':", model.search("search or query"))
    print("'system not database':", model.search("system not database"))
    
    
########################################################################################33
#vsm vector space model
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 31 12:18:33 2025

@author: preth
"""

import math
from collections import Counter

class VectorSpaceModel:
    def __init__(self, docs):
        self.docs = docs
        self.vocab, self.tf_matrix = self._preprocess()
        self.idf = self._compute_idf()
        self.tfidf_matrix = self._compute_tfidf()
    
    def _preprocess(self):
        """Integrated preprocessing"""
        processed = []
        vocab_set = set()
        
        for doc in self.docs:
            tokens = doc.lower().split()
            processed.append(tokens)
            vocab_set.update(tokens)
        
        vocab = sorted(vocab_set)
        
        # Build TF matrix
        tf_matrix = []
        for doc_tokens in processed:
            counts = Counter(doc_tokens)
            tf_row = [counts.get(term, 0) for term in vocab]
            tf_matrix.append(tf_row)
        
        return vocab, tf_matrix
    
    def _compute_idf(self):
        """IDF calculation"""
        N = len(self.tf_matrix)
        idf = []
        for term_idx in range(len(self.vocab)):
            df = sum(1 for doc in self.tf_matrix if doc[term_idx] > 0)
            idf.append(math.log(N / df) if df > 0 else 0)
        return idf
    
    def _compute_tfidf(self):
        """TF-IDF matrix"""
        return [[tf * self.idf[i] for i, tf in enumerate(doc)] 
                for doc in self.tf_matrix]
    
    def _query_to_vector(self, query):
        """Query to TF-IDF vector"""
        query_tf = Counter(query.lower().split())
        return [query_tf.get(term, 0) * self.idf[i] 
                for i, term in enumerate(self.vocab)]
    
    def cosine_similarity(self, v1, v2):
        """Cosine similarity"""
        dot = sum(a * b for a, b in zip(v1, v2))
        mag1 = math.sqrt(sum(a * a for a in v1))
        mag2 = math.sqrt(sum(a * a for a in v2))
        return dot / (mag1 * mag2) if mag1 and mag2 else 0
    
    def jaccard_coefficient(self, v1, v2):
        """Jaccard coefficient for binary vectors"""
        # Convert to binary
        b1 = [1 if x > 0 else 0 for x in v1]
        b2 = [1 if x > 0 else 0 for x in v2]
        
        intersection = sum(a & b for a, b in zip(b1, b2))
        union = sum(a | b for a, b in zip(b1, b2))
        
        return intersection / union if union > 0 else 0
    
    def dice_coefficient(self, v1, v2):
        """Dice coefficient"""
        b1 = [1 if x > 0 else 0 for x in v1]
        b2 = [1 if x > 0 else 0 for x in v2]
        
        intersection = sum(a & b for a, b in zip(b1, b2))
        total = sum(b1) + sum(b2)
        
        return (2 * intersection) / total if total > 0 else 0
    
    def dot_product(self, v1, v2):
        """Simple dot product"""
        return sum(a * b for a, b in zip(v1, v2))
    
    def search(self, query, similarity='cosine', top_k=5):
        """Search with different similarity measures"""
        qvec = self._query_to_vector(query)
        
        similarities = []
        for doc_id, dvec in enumerate(self.tfidf_matrix):
            if similarity == 'cosine':
                sim = self.cosine_similarity(qvec, dvec)
            elif similarity == 'jaccard':
                sim = self.jaccard_coefficient(qvec, dvec)
            elif similarity == 'dice':
                sim = self.dice_coefficient(qvec, dvec)
            elif similarity == 'dot':
                sim = self.dot_product(qvec, dvec)
            else:
                sim = self.cosine_similarity(qvec, dvec)
            
            similarities.append((doc_id, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

# Usage
docs = ["information retrieval system", "machine learning data", "web search engine"]
vsm = VectorSpaceModel(docs)

# Different similarity measures
print("Cosine:", vsm.search("information system", 'cosine'))
print("Jaccard:", vsm.search("information system", 'jaccard'))
print("Dice:", vsm.search("information system", 'dice'))
print("Dot Product:", vsm.search("information system", 'dot'))


##########################################################################################

#bim
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 31 12:24:17 2025

@author: preth
"""

import math

class BinaryIndependenceModel:
    def __init__(self, docs):
        self.docs = docs
        self.vocab, self.binary_matrix = self._preprocess()
        self.N_d = len(self.binary_matrix)  # Total documents
    
    def _preprocess(self):
        """Integrated preprocessing for BIM"""
        vocab_set = set()
        processed = []
        
        for doc in self.docs:
            tokens = set(doc.lower().split())  # Unique tokens per doc
            processed.append(tokens)
            vocab_set.update(tokens)
        
        vocab = sorted(vocab_set)
        
        # Binary matrix
        binary_matrix = []
        for doc_tokens in processed:
            row = [1 if term in doc_tokens else 0 for term in vocab]
            binary_matrix.append(row)
        
        return vocab, binary_matrix
    
    def phase1_estimate(self, query_terms):
        """Phase I: Estimation without relevance information"""
        estimates = {}
        
        for term in query_terms:
            if term not in self.vocab:
                continue
            
            term_idx = self.vocab.index(term)
            
            # d_k = document frequency (number of docs containing term k)
            d_k = sum(1 for doc in self.binary_matrix if doc[term_idx] == 1)
            
            # Phase I formulas (without smoothing first, then with)
            # p_k ≈ 0.5 (assume random chance)
            p_k = 0.5
            
            # q_k ≈ d_k / N_d (simple estimation)
            q_k_simple = d_k / self.N_d if self.N_d > 0 else 0
            
            # q_k with smoothing = (d_k + 0.5) / (N_d + 1)
            q_k = (d_k + 0.5) / (self.N_d + 1)
            
            estimates[term] = {
                'd_k': d_k,
                'p_k': p_k,
                'q_k_simple': q_k_simple,
                'q_k': q_k
            }
        
        return estimates
    
    def phase2_estimate(self, query_terms, relevant_docs):
        """Phase II: Estimation with relevance information"""
        estimates = {}
        N_r = len(relevant_docs)  # Number of relevant docs
        
        for term in query_terms:
            if term not in self.vocab:
                continue
            
            term_idx = self.vocab.index(term)
            
            # r_k = number of relevant docs containing term k
            r_k = sum(1 for doc_id in relevant_docs 
                     if self.binary_matrix[doc_id][term_idx] == 1)
            
            # d_k = total docs containing term k  
            d_k = sum(1 for doc in self.binary_matrix if doc[term_idx] == 1)
            
            # Phase II formulas
            # p_k = r_k / N_r (without smoothing)
            p_k_simple = r_k / N_r if N_r > 0 else 0.5
            
            # p_k = (r_k + 0.5) / (N_r + 1) (with smoothing)
            p_k = (r_k + 0.5) / (N_r + 1)
            
            # q_k = (d_k - r_k) / (N_d - N_r) (without smoothing)
            q_k_simple = (d_k - r_k) / (self.N_d - N_r) if (self.N_d - N_r) > 0 else 0.5
            
            # q_k = (d_k - r_k + 0.5) / (N_d - N_r + 1) (with smoothing)
            q_k = (d_k - r_k + 0.5) / (self.N_d - N_r + 1)
            
            estimates[term] = {
                'r_k': r_k,
                'd_k': d_k,
                'N_r': N_r,
                'p_k_simple': p_k_simple,
                'p_k': p_k,
                'q_k_simple': q_k_simple,
                'q_k': q_k
            }
        
        return estimates
    
    def calculate_rsv(self, doc_id, query_terms, estimates):
        """Calculate Retrieval Status Value"""
        rsv = 0
        
        for term in query_terms:
            if term not in estimates:
                continue
            
            term_idx = self.vocab.index(term)
            p_k = estimates[term]['p_k']
            q_k = estimates[term]['q_k']
            
            # Check if document contains term
            if self.binary_matrix[doc_id][term_idx] == 1:
                # Document contains term: log(p_k / q_k)
                if p_k > 0 and q_k > 0:
                    rsv += math.log(p_k / q_k)
            else:
                # Document doesn't contain term: log((1-p_k) / (1-q_k))
                if p_k < 1 and q_k < 1:
                    rsv += math.log((1 - p_k) / (1 - q_k))
        
        return rsv
    
    def search_phase1(self, query, top_k=5):
        """Phase I search without relevance feedback"""
        query_terms = query.lower().split()
        estimates = self.phase1_estimate(query_terms)
        
        doc_scores = []
        for doc_id in range(self.N_d):
            rsv = self.calculate_rsv(doc_id, query_terms, estimates)
            doc_scores.append((doc_id, rsv))
        
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        return doc_scores[:top_k]
    
    def search_phase2(self, query, relevant_docs, top_k=5):
        """Phase II search with relevance feedback"""
        query_terms = query.lower().split()
        estimates = self.phase2_estimate(query_terms, relevant_docs)
        
        doc_scores = []
        for doc_id in range(self.N_d):
            rsv = self.calculate_rsv(doc_id, query_terms, estimates)
            doc_scores.append((doc_id, rsv))
        
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        return doc_scores[:top_k]
    
    def print_estimates(self, query_terms, relevant_docs=None):
        """Print probability estimates for analysis"""
        if relevant_docs is None:
            print("=== PHASE I ESTIMATES ===")
            estimates = self.phase1_estimate(query_terms)
            for term, est in estimates.items():
                print(f"{term}: d_k={est['d_k']}, p_k={est['p_k']:.3f}, q_k={est['q_k']:.3f}")
        else:
            print("=== PHASE II ESTIMATES ===")
            estimates = self.phase2_estimate(query_terms, relevant_docs)
            for term, est in estimates.items():
                print(f"{term}: r_k={est['r_k']}, d_k={est['d_k']}, N_r={est['N_r']}")
                print(f"      p_k={est['p_k']:.3f}, q_k={est['q_k']:.3f}")

# Usage Example
if __name__ == "__main__":
    docs = [
        "information retrieval system",
        "database search query", 
        "information system database",
        "web search engine",
        "query processing system"
    ]
    
    bim = BinaryIndependenceModel(docs)
    
    query = "information system"
    query_terms = query.split()
    
    print("=== PHASE I (No Relevance Info) ===")
    bim.print_estimates(query_terms)
    results1 = bim.search_phase1(query)
    print(f"Phase I Results: {results1}")
    
    print("\n=== PHASE II (With Relevance Feedback) ===")
    relevant_docs = [0, 2]  # Assume docs 0,2 are relevant
    bim.print_estimates(query_terms, relevant_docs)
    results2 = bim.search_phase2(query, relevant_docs)
    print(f"Phase II Results: {results2}")

# Quick memory formulas
def bim_formulas():
    """Key BIM formulas to remember"""
    print("Phase I (No relevance info):")
    print("  p_k = 0.5")
    print("  q_k = (d_k + 0.5) / (N_d + 1)")
    
    print("\nPhase II (With relevance info):")
    print("  p_k = (r_k + 0.5) / (N_r + 1)")
    print("  q_k = (d_k - r_k + 0.5) / (N_d - N_r + 1)")
    
    print("\nRSV calculation:")
    print("  If term in doc: RSV += log(p_k / q_k)")
    print("  If term not in doc: RSV += log((1-p_k) / (1-q_k))")

######################################################################################3

#without class

#inverted index

# ---------- Build Inverted Index ----------
def build_index(docs):
    index = {}
    for i, doc in enumerate(docs):
        for term in set(doc.lower().split()):
            index.setdefault(term, []).append(i)
    return index


# ---------- Basic Operations ----------
def get(term, index):
    """Get posting list for a term"""
    return index.get(term.lower(), [])


def AND(list1, list2):
    """Intersect two lists"""
    return [x for x in list1 if x in list2]


def OR(list1, list2):
    """Union of two lists"""
    return sorted(set(list1 + list2))


def NOT(posting_list, docs):
    """Complement of posting list"""
    all_docs = list(range(len(docs)))
    return [x for x in all_docs if x not in posting_list]


# ---------- Optimization ----------
def optimize_terms(terms, index, operation="and"):
    """Sort terms by posting list length for optimal processing"""
    term_lengths = [(term, len(get(term, index))) for term in terms]

    if operation == "and":
        # Shortest posting lists first (fewer intersections)
        return [term for term, _ in sorted(term_lengths, key=lambda x: x[1])]
    else:  # OR
        # Longest posting lists first
        return [term for term, _ in sorted(term_lengths, key=lambda x: x[1], reverse=True)]


# ---------- Search ----------
def search(query, docs, index):
    """Optimized boolean search"""
    q = query.lower()

    if " and " in q:
        terms = [t.strip() for t in q.split(" and ")]
        terms = optimize_terms(terms, index, "and")
        result = get(terms[0], index)
        for term in terms[1:]:
            result = AND(result, get(term, index))
            if not result:  # early stop
                break
        return result

    elif " or " in q:
        terms = [t.strip() for t in q.split(" or ")]
        terms = optimize_terms(terms, index, "or")
        result = get(terms[0], index)
        for term in terms[1:]:
            result = OR(result, get(term, index))
        return result

    elif " not " in q:
        pos, neg = q.split(" not ")
        pos_list = get(pos.strip(), index)
        neg_list = get(neg.strip(), index)
        return AND(pos_list, NOT(neg_list, docs))

    else:
        return get(q, index)


# ---------- Usage Example ----------
docs = ["cat dog bird", "dog bird", "cat mouse", "bird eagle", "mouse cat"]
index = build_index(docs)
print("Index:", index)

print("\nQuery: 'cat and bird and dog'")
terms = ["cat", "bird", "dog"]
print("Posting list sizes:")
for term in terms:
    print(f"  {term}: {len(get(term, index))} docs")

optimized = optimize_terms(terms, index, "and")
print(f"Optimized order: {optimized}")
print(f"Result: {search('cat and bird and dog', docs, index)}")

print("\nOR optimization:")
or_optimized = optimize_terms(terms, index, "or")
print(f"OR order: {or_optimized}")

###################################################################################

#bim

import math

# ---------- Preprocessing ----------
def preprocess(docs):
    """Preprocess documents into vocab and binary matrix"""
    vocab_set = set()
    processed = []

    for doc in docs:
        tokens = set(doc.lower().split())  # Unique tokens per doc
        processed.append(tokens)
        vocab_set.update(tokens)

    vocab = sorted(vocab_set)

    # Binary matrix
    binary_matrix = []
    for doc_tokens in processed:
        row = [1 if term in doc_tokens else 0 for term in vocab]
        binary_matrix.append(row)

    return vocab, binary_matrix


# ---------- Phase I (no relevance info) ----------
def phase1_estimate(query_terms, vocab, binary_matrix):
    N_d = len(binary_matrix)
    estimates = {}

    for term in query_terms:
        if term not in vocab:
            continue

        term_idx = vocab.index(term)

        # d_k = number of docs containing term k
        d_k = sum(1 for doc in binary_matrix if doc[term_idx] == 1)

        p_k = 0.5  # random assumption
        q_k_simple = d_k / N_d if N_d > 0 else 0
        q_k = (d_k + 0.5) / (N_d + 1)

        estimates[term] = {
            'd_k': d_k,
            'p_k': p_k,
            'q_k_simple': q_k_simple,
            'q_k': q_k
        }

    return estimates


# ---------- Phase II (with relevance feedback) ----------
def phase2_estimate(query_terms, vocab, binary_matrix, relevant_docs):
    N_d = len(binary_matrix)
    N_r = len(relevant_docs)
    estimates = {}

    for term in query_terms:
        if term not in vocab:
            continue

        term_idx = vocab.index(term)

        # r_k = number of relevant docs containing term
        r_k = sum(1 for doc_id in relevant_docs if binary_matrix[doc_id][term_idx] == 1)

        # d_k = total docs containing term
        d_k = sum(1 for doc in binary_matrix if doc[term_idx] == 1)

        p_k_simple = r_k / N_r if N_r > 0 else 0.5
        p_k = (r_k + 0.5) / (N_r + 1)

        q_k_simple = (d_k - r_k) / (N_d - N_r) if (N_d - N_r) > 0 else 0.5
        q_k = (d_k - r_k + 0.5) / (N_d - N_r + 1)

        estimates[term] = {
            'r_k': r_k,
            'd_k': d_k,
            'N_r': N_r,
            'p_k_simple': p_k_simple,
            'p_k': p_k,
            'q_k_simple': q_k_simple,
            'q_k': q_k
        }

    return estimates


# ---------- RSV Calculation ----------
def calculate_rsv(doc_id, query_terms, estimates, vocab, binary_matrix):
    rsv = 0
    for term in query_terms:
        if term not in estimates:
            continue

        term_idx = vocab.index(term)
        p_k = estimates[term]['p_k']
        q_k = estimates[term]['q_k']

        if binary_matrix[doc_id][term_idx] == 1:
            if p_k > 0 and q_k > 0:
                rsv += math.log(p_k / q_k)
        else:
            if p_k < 1 and q_k < 1:
                rsv += math.log((1 - p_k) / (1 - q_k))

    return rsv


# ---------- Search Functions ----------
def search_phase1(query, vocab, binary_matrix, top_k=5):
    query_terms = query.lower().split()
    estimates = phase1_estimate(query_terms, vocab, binary_matrix)

    doc_scores = []
    for doc_id in range(len(binary_matrix)):
        rsv = calculate_rsv(doc_id, query_terms, estimates, vocab, binary_matrix)
        doc_scores.append((doc_id, rsv))

    doc_scores.sort(key=lambda x: x[1], reverse=True)
    return doc_scores[:top_k]


def search_phase2(query, vocab, binary_matrix, relevant_docs, top_k=5):
    query_terms = query.lower().split()
    estimates = phase2_estimate(query_terms, vocab, binary_matrix, relevant_docs)

    doc_scores = []
    for doc_id in range(len(binary_matrix)):
        rsv = calculate_rsv(doc_id, query_terms, estimates, vocab, binary_matrix)
        doc_scores.append((doc_id, rsv))

    doc_scores.sort(key=lambda x: x[1], reverse=True)
    return doc_scores[:top_k]


# ---------- Printing Helper ----------
def print_estimates(query_terms, vocab, binary_matrix, relevant_docs=None):
    if relevant_docs is None:
        print("=== PHASE I ESTIMATES ===")
        estimates = phase1_estimate(query_terms, vocab, binary_matrix)
        for term, est in estimates.items():
            print(f"{term}: d_k={est['d_k']}, p_k={est['p_k']:.3f}, q_k={est['q_k']:.3f}")
    else:
        print("=== PHASE II ESTIMATES ===")
        estimates = phase2_estimate(query_terms, vocab, binary_matrix, relevant_docs)
        for term, est in estimates.items():
            print(f"{term}: r_k={est['r_k']}, d_k={est['d_k']}, N_r={est['N_r']}")
            print(f"      p_k={est['p_k']:.3f}, q_k={est['q_k']:.3f}")


# ---------- Quick Formulas ----------
def bim_formulas():
    print("Phase I (No relevance info):")
    print("  p_k = 0.5")
    print("  q_k = (d_k + 0.5) / (N_d + 1)")

    print("\nPhase II (With relevance info):")
    print("  p_k = (r_k + 0.5) / (N_r + 1)")
    print("  q_k = (d_k - r_k + 0.5) / (N_d - N_r + 1)")

    print("\nRSV calculation:")
    print("  If term in doc: RSV += log(p_k / q_k)")
    print("  If term not in doc: RSV += log((1-p_k) / (1-q_k))")


# ---------- Usage Example ----------
if __name__ == "_main_":
    docs = [
        "information retrieval system",
        "database search query",
        "information system database",
        "web search engine",
        "query processing system"
    ]

    vocab, binary_matrix = preprocess(docs)

    query = "information system"
    query_terms = query.split()

    print("=== PHASE I (No Relevance Info) ===")
    print_estimates(query_terms, vocab, binary_matrix)
    results1 = search_phase1(query, vocab, binary_matrix)
    print(f"Phase I Results: {results1}")

    print("\n=== PHASE II (With Relevance Feedback) ===")
    relevant_docs = [0, 2]  # Assume docs 0,2 are relevant
    print_estimates(query_terms, vocab, binary_matrix, relevant_docs)
    results2 = search_phase2(query, vocab, binary_matrix, relevant_docs)
    print(f"Phase II Results: {results2}")
    
    
############################################################################3

#vsm

# -- coding: utf-8 --
"""
Created on Sun Aug 31 12:18:33 2025

@author: preth
"""

import math
from collections import Counter

# ------------------ Preprocessing ------------------ #
def preprocess(docs):
    processed = []
    vocab_set = set()
    
    for doc in docs:
        tokens = doc.lower().split()
        processed.append(tokens)
        vocab_set.update(tokens)
    
    vocab = sorted(vocab_set)
    
    # Build TF matrix
    tf_matrix = []
    for doc_tokens in processed:
        counts = Counter(doc_tokens)
        tf_row = [counts.get(term, 0) for term in vocab]
        tf_matrix.append(tf_row)
    
    return vocab, tf_matrix

# ------------------ IDF ------------------ #
def compute_idf(vocab, tf_matrix):
    N = len(tf_matrix)
    idf = []
    for term_idx in range(len(vocab)):
        df = sum(1 for doc in tf_matrix if doc[term_idx] > 0)
        idf.append(math.log(N / df) if df > 0 else 0)
    return idf

# ------------------ TF-IDF ------------------ #
def compute_tfidf(tf_matrix, idf):
    return [[tf * idf[i] for i, tf in enumerate(doc)] for doc in tf_matrix]

# ------------------ Query Vector ------------------ #
def query_to_vector(query, vocab, idf):
    query_tf = Counter(query.lower().split())
    return [query_tf.get(term, 0) * idf[i] for i, term in enumerate(vocab)]

# ------------------ Similarities ------------------ #
def cosine_similarity(v1, v2):
    dot = sum(a * b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a * a for a in v1))
    mag2 = math.sqrt(sum(a * a for a in v2))
    return dot / (mag1 * mag2) if mag1 and mag2 else 0

def jaccard_coefficient(v1, v2):
    b1 = [1 if x > 0 else 0 for x in v1]
    b2 = [1 if x > 0 else 0 for x in v2]
    intersection = sum(a & b for a, b in zip(b1, b2))
    union = sum(a | b for a, b in zip(b1, b2))
    return intersection / union if union > 0 else 0

def dice_coefficient(v1, v2):
    b1 = [1 if x > 0 else 0 for x in v1]
    b2 = [1 if x > 0 else 0 for x in v2]
    intersection = sum(a & b for a, b in zip(b1, b2))
    total = sum(b1) + sum(b2)
    return (2 * intersection) / total if total > 0 else 0

def dot_product(v1, v2):
    return sum(a * b for a, b in zip(v1, v2))

# ------------------ Search ------------------ #
def search(query, vocab, tfidf_matrix, idf, similarity='cosine', top_k=5):
    qvec = query_to_vector(query, vocab, idf)
    similarities = []
    
    for doc_id, dvec in enumerate(tfidf_matrix):
        if similarity == 'cosine':
            sim = cosine_similarity(qvec, dvec)
        elif similarity == 'jaccard':
            sim = jaccard_coefficient(qvec, dvec)
        elif similarity == 'dice':
            sim = dice_coefficient(qvec, dvec)
        elif similarity == 'dot':
            sim = dot_product(qvec, dvec)
        else:
            sim = cosine_similarity(qvec, dvec)
        
        similarities.append((doc_id, sim))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

# ------------------ Usage ------------------ #
docs = ["information retrieval system", "machine learning data", "web search engine"]

# Preprocessing
vocab, tf_matrix = preprocess(docs)
idf = compute_idf(vocab, tf_matrix)
tfidf_matrix = compute_tfidf(tf_matrix, idf)

# Run different similarity searches
print("Cosine:", search("information system", vocab, tfidf_matrix, idf, 'cosine'))
print("Jaccard:", search("information system", vocab, tfidf_matrix, idf, 'jaccard'))
print("Dice:", search("information system", vocab, tfidf_matrix, idf, 'dice'))
print("Dot Product:", search("information system", vocab, tfidf_matrix, idf, 'dot'))












#BOOLEAN MODEL

import pandas as pd
from collections import defaultdict, Counter
from nltk.stem import PorterStemmer

documents = [
    {"doc_id": "D1", "title": "Information requirement", "content": "query considers the user feedback as information requirement to search"},
    {"doc_id": "D2", "title": "Information retrieval", "content": "query depends on the model of information retrieval used"},
    {"doc_id": "D3", "title": "Prediction problem", "content": "Many problems in information retrieval can be viewed as prediction problems"},
    {"doc_id": "D4", "title": "Search", "content": "A search engine is one of applications of information retrieval models"},
    {"doc_id": "D5", "title": "Feedback", "content": "feedback is typically used by the system to modify the query and improve prediction"},
    {"doc_id": "D6", "title": "information retrieval", "content": "ranking in information retrieval algorithms depends on user query"}
]

ps = PorterStemmer()

def preprocess_text(text):
    # Split by whitespace and stem, keeping only alphanumeric words
    tokens = [ps.stem(word.lower()) for word in text.split() if word.isalnum()]
    return tokens

def build_inverted_index(docs, stop_words=None):
    inverted_index = defaultdict(list)
    for doc in docs:
        tokens = preprocess_text(doc["content"])
        if stop_words:
            tokens = [token for token in tokens if token not in stop_words]
        for token in set(tokens):
            inverted_index[token].append(doc["doc_id"])
    return inverted_index

def compute_stop_words(docs):
    all_tokens = []
    for doc in docs:
        all_tokens.extend(preprocess_text(doc["content"]))
    term_freq = Counter(all_tokens)
    stop_words = {term for term, _ in term_freq.most_common(10)}
    return stop_words

def compute_index_size(inverted_index):
    num_terms = len(inverted_index)
    num_postings = sum(len(postings) for postings in inverted_index.values())
    return num_terms, num_postings

def evaluate_boolean_query(query, inverted_index, all_docs):
    tokens = query.lower().split()
    if not tokens:
        return set()

    all_doc_ids = set(doc["doc_id"] for doc in all_docs)
    result = None
    operator = None
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token in ('and', 'or', 'not'):
            operator = token
            i += 1
            continue

        term = ps.stem(token)
        current_docs = set(inverted_index.get(term, []))

        if operator == 'not':
            current_docs = all_doc_ids - current_docs
            operator = None

        if result is None:
            result = current_docs
        elif operator == 'and':
            result = result.intersection(current_docs)
            operator = None
        elif operator == 'or':
            result = result.union(current_docs)
            operator = None

        i += 1

    return result if result is not None else set()

def main():
    initial_index = build_inverted_index(documents)
    initial_terms, initial_postings = compute_index_size(initial_index)

    stop_words = compute_stop_words(documents)
    filtered_index = build_inverted_index(documents, stop_words)
    filtered_terms, filtered_postings = compute_index_size(filtered_index)

    print(f"Stop Words: {stop_words}")
    print(f"\nInitial Index Size: {initial_terms} terms, {initial_postings} postings")
    print(f"Filtered Index Size (without stop words): {filtered_terms} terms, {filtered_postings} postings")

    index_data = [{"Term": term, "Postings": ", ".join(postings)} for term, postings in filtered_index.items()]
    df_index = pd.DataFrame(index_data)
    print("\nInverted Index (without stop words):")
    print(df_index.to_string(index=False))

    sample_queries = [
        "query AND feedback",
        "retrieval OR prediction",
        "search AND NOT engine",
        "information AND retrieval NOT prediction"
    ]

    print("\nBoolean Query Results:")
    query_results = []
    for query in sample_queries:
        result = evaluate_boolean_query(query, filtered_index, documents)
        query_results.append({"Query": query, "Matching Documents": ", ".join(sorted(result)) if result else "None"})

    df_queries = pd.DataFrame(query_results)
    print(df_queries.to_string(index=False))

if __name__ == "__main__":
    main()


##################################################################################################################

import numpy as np
import csv

# Function to manually calculate cosine similarity
def manual_cosine_similarity(vector1, vector2):
    # Convert lists to NumPy arrays for easier computation
    A = np.array(vector1, dtype=float)
    B = np.array(vector2, dtype=float)
    
    # Check if vectors have the same length
    if len(A) != len(B):
        raise ValueError("Vectors must have the same length")
    
    # Compute dot product (A · B)
    dot_product = np.sum(A * B)
    
    # Compute norms (||A|| and ||B||)
    norm_A = np.sqrt(np.sum(A ** 2))
    norm_B = np.sqrt(np.sum(B ** 2))
    
    # Check for zero norms to avoid division by zero
    if norm_A == 0 or norm_B == 0:
        return 0.0  # Return 0 if either vector is a zero vector (undefined case)
    
    # Compute cosine similarity
    similarity = dot_product / (norm_A * norm_B)
    return similarity

######################################################################################


#tf-idf

# -- coding: utf-8 --
"""
Created on Sun Aug 31 11:53:28 2025

@author: preth
"""

# ------------------ Build Term-Document Matrix ------------------ #
def build_matrix(documents):
    all_terms = set()
    processed_docs = []
    
    for doc in documents:
        tokens = doc.lower().split()
        processed_docs.append(tokens)
        all_terms.update(tokens)
    
    vocab = sorted(all_terms)
    
    # Build term-document matrix
    term_doc_matrix = []
    for term in vocab:
        row = []
        for doc_tokens in processed_docs:
            row.append(1 if term in doc_tokens else 0)
        term_doc_matrix.append(row)
    
    return vocab, term_doc_matrix


# ------------------ Term Vector ------------------ #
def get_term_vector(term, vocab, term_doc_matrix, num_docs):
    term = term.lower()
    if term not in vocab:
        return [0] * num_docs
    
    term_idx = vocab.index(term)
    return term_doc_matrix[term_idx]


# ------------------ Boolean Operations ------------------ #
def boolean_and(vec1, vec2):
    return [a & b for a, b in zip(vec1, vec2)]

def boolean_or(vec1, vec2):
    return [a | b for a, b in zip(vec1, vec2)]

def boolean_not(vec):
    return [1 - x for x in vec]


# ------------------ Search ------------------ #
def search(query, vocab, term_doc_matrix, documents):
    query = query.lower().strip()
    num_docs = len(documents)
    
    # Single term
    if ' ' not in query:
        result_vector = get_term_vector(query, vocab, term_doc_matrix, num_docs)
    
    # AND operation
    elif ' and ' in query:
        terms = [t.strip() for t in query.split(' and ')]
        result_vector = get_term_vector(terms[0], vocab, term_doc_matrix, num_docs)
        for term in terms[1:]:
            term_vec = get_term_vector(term, vocab, term_doc_matrix, num_docs)
            result_vector = boolean_and(result_vector, term_vec)
    
    # OR operation
    elif ' or ' in query:
        terms = [t.strip() for t in query.split(' or ')]
        result_vector = get_term_vector(terms[0], vocab, term_doc_matrix, num_docs)
        for term in terms[1:]:
            term_vec = get_term_vector(term, vocab, term_doc_matrix, num_docs)
            result_vector = boolean_or(result_vector, term_vec)
    
    # NOT operation
    elif ' not ' in query:
        parts = query.split(' not ')
        pos_term = parts[0].strip()
        neg_term = parts[1].strip()
        
        pos_vec = get_term_vector(pos_term, vocab, term_doc_matrix, num_docs)
        neg_vec = get_term_vector(neg_term, vocab, term_doc_matrix, num_docs)
        neg_vec = boolean_not(neg_vec)
        result_vector = boolean_and(pos_vec, neg_vec)
    
    else:
        result_vector = [0] * num_docs
    
    # Return document IDs where result is 1
    return [i for i, val in enumerate(result_vector) if val == 1]


# ------------------ Print Matrix ------------------ #
def print_matrix(vocab, term_doc_matrix, documents):
    print("Term-Document Matrix:")
    print("Terms\\Docs", end="")
    for i in range(len(documents)):
        print(f"\tD{i}", end="")
    print()
    
    for i, term in enumerate(vocab):
        print(f"{term:<10}", end="")
        for val in term_doc_matrix[i]:
            print(f"\t{val}", end="")
        print()


# ------------------ Usage Example ------------------ #
if _name_ == "_main_":
    docs = [
        "information retrieval system",
        "database search query",
        "information system database",
        "web search engine",
        "query processing system"
    ]
    
    vocab, term_doc_matrix = build_matrix(docs)
    print_matrix(vocab, term_doc_matrix, docs)
    
    print("\nSearch Results:")
    print("'information':", search("information", vocab, term_doc_matrix, docs))
    print("'information and system':", search("information and system", vocab, term_doc_matrix, docs))
    print("'search or query':", search("search or query", vocab, term_doc_matrix, docs))
    print("'system not database':", search("system not database", vocab, term_doc_matrix, docs))
    
    
    
#############################################################################################33
