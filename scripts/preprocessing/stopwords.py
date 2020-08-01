negations = ['no', 'cannot', 'not', 'none', 'nothing', 'nowhere',
             'neither', 'nor', 'nobody', 'hardly', 'scarcely', 'barely']

""" taken from https://gist.github.com/sebleier/554280 """

neg_stopwords = ["aren't", "couldn't", "hasn't", "didn't", "doesn't", "don't", "hadn't", "isn't", "haven't", "mightn't", "mustn't", "needn't", "shan't", "wasn't", "weren't", "won't", "shouldn't",
                 "wouldn't", "can't", "ain't", "couldnt", "arent", "wasnt", "werent", "cant", "wont", "cant", "wouldnt", "couldnt", "hasnt", 'n\'t']

stopwords = ["a", "about", "above", "after", "again", "against", "ain", "all", "am", "an", "and", "any", "are", "aren",  "as", "at", "be", "because", "been", "before", "being", "below", "between",
             "both", "but", "by", "can", "couldn",  "d", "did", "didn",  "do", "does", "doesn", "doing", "down", "during", "each", "few", "for", "from", "further",
             "had", "hadn",  "has", "hasn",  "have", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "isn",
             "it", "it's", "its", "itself", "just", "ll", "m", "ma", "me", "mightn",  "more", "most", "mustn",  "my", "myself", "needn",  "no", "nor", "not", "now", "o",
             "of", "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "re", "s", "same", "shan",  "she", "she's", "should", "should've", "shouldn",
             "so", "some", "such", "t", "than", "that", "that'll", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up",
             "ve", "very", "was", "wasn",  "we", "were", "weren",  "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with", "won",  "wouldn",  "y",
             "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves", "could", "he'd", "he'll", "he's", "here's", "how's", "i'd", "i'll", "i'm", "i've", "let's", "ought", "she'd",
             "she'll", "that's", "there's", "they'd", "they'll", "they're", "they've", "we'd", "we'll", "we're", "we've", "what's", "when's", "where's", "who's", "why's", "would", "able", "abst",
             "according", "accordingly", "across", "actually", "adj", "affected", "affects", "afterwards", "ah", "almost", "alone", "along", "already", "also", "although", "always",
             "among", "amongst", "another", "anybody", "anyhow", "anymore", "anyone", "anything", "anyway", "anyways", "anywhere", "arise", "around", "aside",
             "ask", "asking", "auth", "available", "away", "awfully", "b", "became", "become", "becomes", "becoming", "beforehand", "beginnings", "begins", "behind", "believe",
             "beside", "besides", "beyond", "biol", "brief", "c", "ca", "came", "cannot",  "cause", "causes", "certain", "certainly", "co", "com", "come", "comes", "contain", "containing",
             "contains", "different", "done", "downwards", "e", "ed", "edu", "eg", "eight", "either", "else", "elsewhere", "enough", "especially",
             "et", "etc", "even", "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex", "except", "f", "far", "ff", "fifth", "first", "five", "followed", "following", "follows",
             "former", "formerly", "forth", "four", "furthermore", "g", "gave", "get", "gets", "getting", "give", "given", "gives", "go", "goes", "gone", "got", "gotten", "h", "happens",
             "hardly", "hed", "hence", "hereafter", "hereby", "herein", "heres", "hereupon", "hes", "hi", "hid", "hither", "home", "howbeit", "however",
             "id", "ie", "im", "immediate", "inc", "indeed", "instead", "inward", "itd", "it'll", "j", "k", "keep",
             "keeps", "kept", "kg", "km", "know", "known", "knows", "l", "largely", "last", "lately", "later", "latter", "latterly", "least", "less", "lest", "let", "lets", "like", "liked", "likely",
             "little", "'ll", "look", "looking", "looks", "ltd", "made", "mainly", "makes", "many", "may", "maybe", "mean", "meanwhile", "merely", "mg", "might",
             "ml", "moreover", "mostly", "mr", "mrs", "much", "must", "n", "na", "name", "namely", "nd", "near", "nearly", "necessary", "need", "needs", "neither", "never",
             "nevertheless", "new", "next", "nine", "nobody", "non", "none", "nonetheless", "noone", "normally", "nos", "nothing", "nowhere", "obtained", "obviously", "often",
             "oh", "ok", "okay", "old", "one", "ones", "onto", "ord", "others", "otherwise", "outside", "overall", "pages", "particular", "particularly", "per",
             "perhaps", "placed", "please", "plus", "possible", "potentially", "pp", "predominantly",
             "primarily", "probably", "promptly", "provides", "q", "que", "quite", "qv", "r", "ran", "rather", "rd", "really", "recently", "ref", "refs",
             "regarding", "regardless", "regards", "relatively", "respectively", "resulted", "resulting", "results", "right", "run", "said", "saw", "say", "saying", "says",
             "sec", "section", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", "selves", "sent", "seven", "several", "shall", "shes", "showed", "shown", "showns",
             "shows", "significantly", "similarly", "since", "six", "somebody", "somehow", "someone", "somethan", "something", "sometime", "sometimes", "somewhat",
             "somewhere", "soon", "sorry", "specifically", "specified", "specify", "specifying", "still", "sub", "successfully", "sup", "sure",
             "take", "taken", "taking", "tell", "tends", "th", "thank", "thanks", "thanx", "thats", "that've", "thence", "thereafter", "thereby", "thered", "therefore", "therein", "there'll", "therere",
             "theres", "thereto", "thereupon", "there've", "theyd", "theyre", "think", "thou", "though", "thoughh", "throug", "throughout", "thru", "thus", "til", "together", "took", "toward",
             "towards", "tried", "tries", "truly", "try", "trying", "ts", "twice", "two", "u", "un", "unfortunately", "unless", "unlikely", "unto", "upon", "ups", "us", "use", "used", "useful",
             "uses", "using", "usually", "v", "value", "various", "'ve", "via", "viz", "vol", "vols", "vs", "w", "want", "wants", "way", "welcome", "went",  "whatever", "what'll",
             "whats", "whence", "whenever", "whereafter", "whereas", "whereby", "wherein", "wheres", "whereupon", "wherever", "whether", "whither", "whod", "whoever", "whole", "who'll", "whomever", "whos", "whose",
             "willing", "wish", "within", "without", "www", "x", "yes", "yet", "youd", "youre", "z", "zero", "a's",  "allow", "allows", "apart",
             "appear", "appreciate", "appropriate", "associated", "best", "better", "c'mon", "c's",  "changes", "clearly", "concerning", "consequently", "consider", "considering",
             "corresponding", "course", "currently", "definitely", "described", "despite", "entirely", "exactly", "example", "going", "greetings", "hello", "help", "hopefully", "ignored",
             "inasmuch", "indicate", "indicated", "indicates", "inner", "insofar", "it'd", "keep", "keeps", "novel", "presumably", "reasonably", "second", "secondly", "sensible", "serious",
             "seriously", "sure", "t's", "third", "thorough", "thoroughly", "three", "well", "wonder", "a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all",
             "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "an", "and", "another", "any", "anyhow", "anyone", "anything",
             "anyway", "anywhere", "are", "around", "as", "at", "be", "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside",
             "besides", "between", "beyond", "both", "but", "by", "can", "cannot",  "co", "could", "de", "do", "done",
             "down", "during", "each", "eg", "eight", "either", "else", "elsewhere", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except",
             "few", "fifteen", "fify", "first", "five", "for", "former", "formerly", "four", "from", "further", "get", "give", "go", "had", "has",
             "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "ie", "if", "in", "inc",
             "indeed", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "more",
             "moreover", "most", "mostly", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing",
             "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own", "per", "perhaps",
             "please", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "since", "six", "so", "some",
             "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "take", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there",
             "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to",
             "together", "too", "toward", "towards", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when",
             "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why",
             "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q",
             "r", "s", "t", "u", "v", "w", "x", "y", "z", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "co", "op",
             "research-articl", "pagecount", "cit", "ibid", "les", "le", "au", "que", "est", "vol", "el", "los", "pp", "u201d", "well-b", "http", "volumtype", "0o", "0s", "3a", "3b", "3d", "6b",
             "6o", "a1", "a2", "a3", "a4", "ab", "ac", "ad", "ae", "af", "ag", "aj", "al", "an", "ao", "ap", "ar", "av", "aw", "ax", "az", "b1", "b2", "b3", "ba", "bc", "bd", "be", "bi", "bj", "bk", "bl",
             "bn", "bp", "br", "bs", "bt", "bu", "bx", "c1", "c2", "c3", "cc", "cd", "ce", "cf", "cg", "ch", "ci", "cj", "cl", "cm", "cn", "cp", "cq", "cr", "cs", "ct", "cu", "cv", "cx", "cy", "cz", "d2", "da",
             "dc", "dd", "de", "df", "di", "dj", "dk", "dl", "do", "dp", "dr", "ds", "dt", "du", "dx", "dy", "e2", "e3", "ea", "ec", "ed", "ee", "ef", "ei", "ej", "el", "em", "en", "eo", "ep", "eq", "er", "es",
             "et", "eu", "ev", "ex", "ey", "f2", "fa", "fc", "ff", "fi", "fj", "fl", "fn", "fo", "fr", "fs", "ft", "fu", "fy", "ga", "ge", "gi", "gj", "gl", "go", "gr", "gs", "gy", "h2", "h3", "hh", "hi", "hj",
             "ho", "hr", "hs", "hu", "hy", "i", "i2", "i3", "i4", "i6", "i7", "i8", "ia", "ib", "ic", "ie", "ig", "ih", "ii", "ij", "il", "in", "io", "ip", "iq", "ir", "iv", "ix", "iy", "iz", "jj", "jr", "js",
             "jt", "ju", "ke", "kg", "kj", "km", "ko", "l2", "la", "lb", "lc", "lf", "lj", "ln", "lo", "lr", "ls", "lt", "m2", "ml", "mn", "mo", "ms", "mt", "mu", "n2", "nc", "nd", "ne", "ng", "ni", "nj", "nl",
             "nn", "nr", "ns", "ny", "oa", "ob", "oc", "od", "of", "og", "oi", "oj", "ol", "om", "on", "oo", "oq", "or", "os", "ot", "ou", "ow", "oz", "p1", "p2", "p3", "pc", "pd", "pe", "pf", "ph",
             "pi", "pj", "pk", "pl", "pm", "pn", "po", "pq", "pr", "ps", "pt", "pu", "py", "qj", "qu", "r2", "ra", "rc", "rd", "rf", "rh", "ri", "rj", "rl", "rm", "rn", "ro", "rq", "rr", "rs", "rt", "ru", "rv",
             "ry", "s2", "sa", "sc", "sd", "se", "sf", "si", "sj", "sl", "sm", "sn", "sp", "sq", "sr", "ss", "st", "sy", "sz", "t1", "t2", "t3", "tb", "tc", "td", "te", "tf", "th", "ti", "tj", "tl", "tm", "tn",
             "tp", "tq", "tr", "ts", "tt", "tv", "tx", "ue", "ui", "uj", "uk", "um", "un", "uo", "ur", "ut", "va", "wa", "vd", "wi", "vj", "vo", "wo", "vq", "vt", "vu", "x1", "x2", "x3", "xf", "xi", "xj", "xk",
             "xl", "xn", "xo", "xs", "xt", "xv", "xx", "y2", "yj", "yl", "yr", "ys", "yt", "zi", "zz"]