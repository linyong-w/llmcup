from functools import wraps

from flask import make_response, request

def auth_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization

        
        # if auth and auth.username == current_app.config["SITE_USER"] and auth.password == current_app.config["SITE_PASS"]:
        if auth and auth.username == "puspaadm" and auth.password == "petronas@123":
        
            return f(*args, **kwargs)
        return make_response("<body><h1>Unauthorized access</h1><p>You are denied access to this resource. Please login to proceed </p></body>", 401, {'WWW-Authenticate': 'Basic realm="Login required!"'})

    return decorated