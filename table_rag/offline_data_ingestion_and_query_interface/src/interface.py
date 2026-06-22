from flask import Flask, request, jsonify
from service import process_tablerag_request

app = Flask(__name__)

@app.route('/get_tablerag_response', methods=['POST'])
def get_tablerag_response():
    json_body = request.get_json()
    if not json_body or 'query' not in json_body or 'table_name_list' not in json_body:
        return jsonify({'error': 'Invalid input'}), 400

    query = json_body['query']
    table_name_list = json_body['table_name_list']

    res_dict = process_tablerag_request(table_name_list, query)
    
    return jsonify(res_dict)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)