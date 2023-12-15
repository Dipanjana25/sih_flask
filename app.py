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

    locs=flair_token.tokenise_loc(query)
    print(locs)
    mp=defaultdict(list)
    for token in locs:
        mp[str(token[0])]=ngram_fuzzy_search.fuzzy_search(token[0], 60)
    print(mp)
    return jsonify({
        'loc_tokens': locs,
        'fuzzy_matches': mp,
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
            ngram_fuzzy_search.delete(request.json['locs'])
            return jsonify({"status": "success"}), 200
        except:
            return jsonify({"status": "not success"}), 400

    return jsonify({"status": "invalid method"})


if __name__== "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)