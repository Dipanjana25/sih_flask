from flask import Flask, request, abort, jsonify

import flair_token
import fuzzy_search

app = Flask(__name__)
tokenjson = "output/token.json"
resultsjson = "output/results.json"

@app.route('/api/processquery', methods=['POST'])
async def processquery():
    if not request.json or not 'query' in request.json:
        abort(400)
    query=request.json['query']

    loc=await flair_token.tokenise_loc(query)
    possible_loc_tokens=await flair_token.get_possible_loc_tokens(query)
    
    mp={}
    for token in possible_loc_tokens:
            mp[token[0]]=[]
            mp[token[0]]=await fuzzy_search.closest(token[0])
    
    return jsonify({
        'flair_loc_tokens': loc,
        'possible_loc_tokens': possible_loc_tokens,
        'fuzzy_sdx': mp,
    })

if __name__== "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)