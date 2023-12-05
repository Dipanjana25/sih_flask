from flask import Flask,render_template,request,jsonify
import json
app = Flask(__name__)

import flair_token
import fuzzy_search
tokenjson = "output/token.json"
resultsjson = "output/results.json"

@app.route('/', methods = ['POST','GET'])
def home():
    if request.method =="POST":
        sentence = request.form['query']
        try:
            token  = flair_token.handle_click(sentence)
            print(token)

            with open(tokenjson, 'w') as json_file:
                json.dump(token, json_file)
            
            results = []
            for i in token:
                results.append(fuzzy_search.closest(i))
            
            # print(results)
            with open(resultsjson, 'w') as json_file:
                json.dump(results, json_file)

            token_json = json.dumps(token)

            results_json = json.dumps(results)
            # return render_template('base.html',token=token_json,results=results_json)
            return jsonify(token)
        except:
            return render_template('base.html')
    else:
        return render_template('base.html')
 


if __name__== "__main__":
    app.run(debug=True)