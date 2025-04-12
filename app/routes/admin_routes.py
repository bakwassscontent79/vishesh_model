from flask import Blueprint, render_template, redirect, url_for

admin_routes = Blueprint('admin_routes', __name__)

# Root route → redirect to heart UI
@admin_routes.route("/")
def dashboard_redirect():
    return redirect(url_for("admin_routes.heart_ui"))

# Main heart disease UI
@admin_routes.route("/admin/heart-disease/")
def heart_ui():
    return render_template("heart_disease.html")
