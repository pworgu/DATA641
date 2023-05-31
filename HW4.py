import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
def tokenize(text):
return [token.lower() for token in word_tokenize(text)]
source = "Fodd bynnag, yr wyf yn ystyried bod iaith hiliol, iaith sy’n gwahaniaethu ar sail rhyw neu ar unrhyw sail arall, a honiadau yn erbyn Aelodau, yn peri tramgwydd."
reference = "However, I consider that racist, sexist or other discriminatory language, and allegations against Members, are offensive."
candidates = [
"However, I consider racist language, sexist or other discrimination, and allegations against Members to be offensive.",
"However, I regard racist language, language that discriminates on the basis of sex or on any other grounds, and allegations against Members, as offensive.",
"Racist Members consider that discriminatory allegations as language are the basis of offensive sexist allegations, however.",
"Allegations against members are offensive."
]
reference_tokens = tokenize(reference)
candidate_tokens = [tokenize(candidate) for candidate in candidates]
for i, candidate in enumerate(candidate_tokens):
print(f"Candidate {i + 1}:")
for n in range(1, 4):
weights = [1.0 if i == n - 1 else 0.0 for i in range(4)]
bleu_score = nltk.translate.bleu_score.sentence_bleu([reference_tokens], candidate, weights=weights)
print(f"{n}-gram precision: {bleu_score:.3f}")
print()
