from flask import Flask,render_template,request

app = Flask(__name__)

import flair_token
import fuzzy_search


@app.route('/', methods = ['POST','GET'])
def home():
    if request.method =="POST":
        sentence = request.form['query']
        try:
            token  = flair_token.handle_click(sentence)
            results = []
            for i in token:
                results.append(fuzzy_search.closest(i))

            print(results)
            return render_template('base.html',tokens=results)
        except:
            return render_template('base.html')
    else:
        return render_template('base.html')
 


if __name__== "__main__":
    app.run(debug=True)