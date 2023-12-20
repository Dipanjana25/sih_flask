from flask import Flask, request, abort, jsonify
from collections import defaultdict
import flair_token
import ngram_fuzzy_search

app = Flask(__name__)

@app.route('/api/processquery', methods=['POST'])
def processquery():
    if not request.json or not 'query' in request.json:
        abort(400)
    query=request.json['query']

    locs1=flair_token.tokenise_loc(query)
    locs2=flair_token.tokenise_loc_2(query)
    # locs1=[i for i in locs1 if i[1]=='LOC']
    # print(locs)
    mp1=defaultdict(list)
    mp2=defaultdict(list)
    for token in locs1:
        mp1[str(token[0])]=ngram_fuzzy_search.fuzzy_search(token[0], 60)
    for token in locs2:
        mp2[str(token[0])]=ngram_fuzzy_search.fuzzy_search(token[0], 60)
    # print(mp)
    return jsonify({
        'model_1_loc': locs1,
        'model_2_loc': locs2,
        'fuzzy_matches_1': mp1,
        'fuzzy_matches_2': mp2
    })

@app.route('/api/ngindex', methods=['POST', 'PUT', 'DELETE'])
def update():
    if not request.json:
        abort(400)

    if request.method=='POST':
        if not 'loc' in request.json:
            abort(400)
        try:
            ngram_fuzzy_search.insert(request.json['loc'])
            return jsonify({"status": "success"}), 200
        except:
            return jsonify({"status": "not success"}), 400

    elif request.method=='PUT':
        if 'old_loc' not in request.json:
            abort(400)
        elif 'new_loc' not in request.json:
            abort(400)
        try:
            ngram_fuzzy_search.update(request.json['old_loc'], request.json['new_loc'])
            return jsonify({"status": "success"}), 200
        except:
            return jsonify({"status": "not success"}), 400
    
    elif request.method=='DELETE':
        if 'loc' not in request.json:
            abort(400)
        try:
            ngram_fuzzy_search.delete(request.json['loc'])
            return jsonify({"status": "success"}), 200
        except:
            return jsonify({"status": "not success"}), 400

    return jsonify({"status": "invalid method"})


if __name__== "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)