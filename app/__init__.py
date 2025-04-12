from flask import Flask

app = Flask(__name__)

# Import blueprints
from app.services.heart_disease.main import heart_disease_bp
from app.routes.admin_routes import admin_routes  # ✅ Add this

# Register blueprints
app.register_blueprint(heart_disease_bp)
app.register_blueprint(admin_routes)  # ✅ Register admin_routes for "/" and /admin/heart-disease/