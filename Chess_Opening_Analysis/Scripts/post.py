# post processing




# jonathan functies vergelijk

# while x < game size:
#   ngram frequencies maken van al die games
#   cosine similarities uitrekenen
#   avg cosine similarity
#   x_values.append(avg cos)
#   y_values.append(gamesize)
#   game size +1

# verschillende lengtes append in games



def cosine_similarity(freqs1, freqs2):
        boven = 0
        asq = 0
        bsq = 0
        for x in freqs1:
            if x in freqs2:
                boven += freqs1[x] * freqs2[x]
            asq += (freqs1[x])**2
        for y in freqs2:
            bsq += (freqs2[y])**2
        cos = boven / math.sqrt(asq*bsq)
        return cos

x_values = []
y_values = []

# voor verschillende sizes van zetten (tot hoeveel zetten in het potje we de similarity berekenen)
# zet sizes in x values
# zet uitkomsten van avg similarity in y values (voor zoeken omslagpunt)
for x in sizes:

    # games lijsten krijgen we dmv van functie van jonathan die ngrams maakt
    games = []

    cosines = []
    
    matrix = []
    for game in games:
        rows = []
        for compare in games:
            cos = cosine_similarity(game, compare)
            rows.append(cos)
            cosines.append(cos)
        matrix.append(rows)

    average = cosines.mean()
    y_values.append(cosines)
