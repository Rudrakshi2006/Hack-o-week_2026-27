# Whiteboard REST API using Flask

## Project Overview

This project is a simple whiteboard application built using **Flask**. The main idea is to create a drawing board where some sections are **restricted** and cannot be edited by the user.

The whiteboard is divided into three parts:

* **Header** (Restricted)
* **Drawing Area (Canvas)** (Editable)
* **Footer** (Restricted)

If the user tries to draw in the header or footer section, a popup message is displayed saying **"You cannot edit this section!"** Only the middle drawing area can be edited.

## Problem Statement

In many real-world applications, certain parts of a document or whiteboard, such as the company header, logo, or footer, should remain unchanged while allowing users to edit the main content.

This project demonstrates how to protect specific areas from being modified while still allowing drawing in the editable region.

## Solution

The application uses a single HTML Canvas where the header and footer are marked as restricted areas. Whenever the user starts drawing, the application checks the mouse position.

* If the mouse is inside the **Header** or **Footer**, drawing is blocked and a popup message is shown.
* If the mouse is inside the **Canvas (Drawing Area)**, drawing is allowed.

Flask is used to serve the web application, and in future versions, REST API endpoints will be added to save, load, update, and delete whiteboard data.

## Technologies Used

* Python
* Flask
* HTML
* CSS
* JavaScript
* HTML5 Canvas

## Features (Current)

* Header section
* Footer section
* Editable drawing canvas
* Restricted editing in header and footer
* Popup message for restricted areas

## Future Enhancements

* REST API for saving and loading drawings
* Company logo restriction
* Eraser tool
* Text tool
* Clear whiteboard option
* Store whiteboard data in a JSON file or database
