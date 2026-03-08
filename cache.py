from sklearn.metrics.pairwise import cosine_similarity

class SemanticCache:

    def __init__(self, threshold=0.85):

        self.entries = []
        self.threshold = threshold

        self.hit_count = 0
        self.miss_count = 0

    def lookup(self, query_embedding):

        for entry in self.entries:

            sim = cosine_similarity(
                [query_embedding],
                [entry["embedding"]]
            )[0][0]

            if sim > self.threshold:

                self.hit_count += 1

                return True, entry, sim

        self.miss_count += 1

        return False, None, None


    def store(self, query, embedding, result, cluster):

        self.entries.append({
            "query": query,
            "embedding": embedding,
            "result": result,
            "cluster": cluster
        })


    def stats(self):

        total = self.hit_count + self.miss_count

        rate = self.hit_count / total if total else 0

        return {
            "total_entries": len(self.entries),
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": rate
        }


    def clear(self):

        self.entries = []
        self.hit_count = 0
        self.miss_count = 0