# PixelArt Studio

**PixelArt Studio** is an interactive pixel art creation platform that allows users to create, style, and export pixel art. It features advanced neural style transfer capabilities, enabling users to apply artistic styles to their creations.

## Features

- **Interactive Drawing Tools**: Intuitive interface for creating pixel art with customizable canvas sizes.
- **Neural Style Transfer**: Apply artistic styles using PyTorch and VGG19.
- **Responsive UI**: Smooth performance across devices with custom animations and effects.
- **Image Processing**: Import/export support with smart scaling and pixel-perfect rendering.

## Setup Instructions

### Prerequisites

- Node.js (v14 or later)
- Python (v3.8 or later)
- pip (Python package manager)

---

### Frontend Setup

1. Navigate to the frontend directory:

    ```bash
    cd my-pixel-art
    ```

2. Install dependencies:

    ```bash
    npm install
    ```

3. Start the frontend server:

    ```bash
    npm start
    ```

The frontend will be available at [http://localhost:3000](http://localhost:3000).

---

### Backend Setup

1. Navigate to the backend directory:

    ```bash
    cd backend
    ```

2. Create a virtual environment:

    ```bash
    python -m venv venv
    ```

3. Activate the virtual environment:

    - On Windows:

        ```bash
        venv\Scripts\activate
        ```

    - On macOS/Linux:

        ```bash
        source venv/bin/activate
        ```

4. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

5. Start the backend server:

    ```bash
    python complete_neural_style.py
    ```

The backend will be available at [http://localhost:5001](http://localhost:5001).

---

## Usage

1. Open your browser and navigate to [http://localhost:3000](http://localhost:3000).
2. Use the drawing tools to create your pixel art.
3. Apply styles using the neural style transfer feature.
4. Export your artwork in various formats.

---

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any improvements or bug fixes.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contact

For questions or support, please contact **steviehu95@example.com**.
