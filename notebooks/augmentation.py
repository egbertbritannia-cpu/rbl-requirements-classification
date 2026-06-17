import random
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize

class RequirementAugmenter:
    def __init__(self):
        # Trigger downloads if they haven't been downloaded yet
        try:
            nltk.data.find('corpora/wordnet')
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            nltk.download('wordnet')
            nltk.download('punkt')
            nltk.download('averaged_perceptron_tagger')
            nltk.download('omw-1.4')
        
        self.stop_words = set(nltk.corpus.stopwords.words('english')) if 'stopwords' in dir(nltk.corpus) else set()
        # Fallback basic stopwords if nltk stopwords are missing
        if not self.stop_words:
            self.stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'to', 'in', 'on', 'with', 'for', 'and', 'or'}
            
        self.srs_phrases = [
            "the system shall",
            "the application must",
            "must be able to",
            "should ensure that",
            "is required to"
        ]

    def _get_synonyms(self, word):
        synonyms = set()
        for syn in wordnet.synsets(word):
            for l in syn.lemmas():
                synonym = l.name().replace("_", " ").replace("-", " ").lower()
                synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
                synonyms.add(synonym) 
        if word in synonyms:
            synonyms.remove(word)
        return list(synonyms)

    def synonym_replacement(self, text, n=1):
        """1. Substitutes selected non-stopwords with semantically equivalent alternatives using WordNet."""
        words = word_tokenize(text)
        new_words = words.copy()
        random_word_list = list(set([word for word in words if word.lower() not in self.stop_words and word.isalpha()]))
        random.shuffle(random_word_list)
        
        num_replaced = 0
        for random_word in random_word_list:
            synonyms = self._get_synonyms(random_word.lower())
            if len(synonyms) >= 1:
                synonym = random.choice(list(synonyms))
                new_words = [synonym if word == random_word else word for word in new_words]
                num_replaced += 1
            if num_replaced >= n: 
                break
        
        return ' '.join(new_words)

    def phrase_insertion(self, text):
        """2. Adds common requirement-related expressions."""
        words = word_tokenize(text)
        if len(words) == 0: return text
        
        # Determine if it already starts with an SRS phrase
        lower_text = text.lower()
        has_phrase = any(lower_text.startswith(p) for p in self.srs_phrases)
        
        if not has_phrase:
            phrase = random.choice(self.srs_phrases)
            return phrase + " " + text
        return text # If it already has one, don't insert to avoid grammar mess

    def text_simplification(self, text):
        """3. Removes verbose phrases to focus on core semantics."""
        words = word_tokenize(text)
        if len(words) < 5: return text
        
        # A simple heuristic: drop some extremely common verbose fillers or adjectives
        fillers = {'basically', 'essentially', 'in order to', 'so as to', 'actually', 'simply'}
        simplified = [w for w in words if w.lower() not in fillers]
        
        # We can also drop a random stopword to simulate simplification of verbosity
        stopword_indices = [i for i, w in enumerate(simplified) if w.lower() in self.stop_words]
        if stopword_indices:
            drop_idx = random.choice(stopword_indices)
            simplified.pop(drop_idx)
            
        return ' '.join(simplified)

    def random_word_swap(self, text):
        """4. Introduces minor structural variations by swapping two random words."""
        words = word_tokenize(text)
        if len(words) < 4: return text
        
        idx1, idx2 = random.sample(range(len(words)), 2)
        words[idx1], words[idx2] = words[idx2], words[idx1]
        return ' '.join(words)

    def random_word_deletion(self, text, p=0.1):
        """5. Randomly deletes words with probability p."""
        words = word_tokenize(text)
        if len(words) < 4: return text

        new_words = []
        for word in words:
            if random.uniform(0, 1) > p:
                new_words.append(word)
                
        if len(new_words) == 0:
            return words[random.randint(0, len(words)-1)]
            
        return ' '.join(new_words)

    def augment(self, text, strategy='random'):
        """Applies one of the 5 strategies."""
        methods = [
            self.synonym_replacement,
            self.phrase_insertion,
            self.text_simplification,
            self.random_word_swap,
            self.random_word_deletion
        ]
        
        if strategy == 'random':
            method = random.choice(methods)
            return method(text)
        else:
            return methods[strategy](text)

if __name__ == "__main__":
    # Test
    aug = RequirementAugmenter()
    sentence = "The application should be easy to use."
    print("Original:", sentence)
    print("1. Synonym:  ", aug.synonym_replacement(sentence))
    print("2. Phrase:   ", aug.phrase_insertion(sentence))
    print("3. Simplify: ", aug.text_simplification(sentence))
    print("4. Swap:     ", aug.random_word_swap(sentence))
    print("5. Deletion: ", aug.random_word_deletion(sentence))
