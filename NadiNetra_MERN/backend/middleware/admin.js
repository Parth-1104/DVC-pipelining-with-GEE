const jwt = require("jsonwebtoken")
const {JWT_ADMIN_SECRET} = require("../config")

const adminMiddleware = async (req, res, next) => {
    try {
        const token = req.headers.token;
        
        if (!token) {
            return res.status(401).json({
                message: "No token provided"
            });
        }

        const decoded = jwt.verify(token, JWT_ADMIN_SECRET);
        
        
        req.userId = decoded.id;
        
        next();
    } catch (error) {
        if (error.name === 'JsonWebTokenError') {
            return res.status(403).json({
                message: "Invalid token"
            });
        }
        
        if (error.name === 'TokenExpiredError') {
            return res.status(401).json({
                message: "Token expired"
            });
        }

        console.error("Authentication error:", error);
        res.status(500).json({
            message: "Authentication failed"
        });
    }
}

module.exports = {
    adminMiddleware
}