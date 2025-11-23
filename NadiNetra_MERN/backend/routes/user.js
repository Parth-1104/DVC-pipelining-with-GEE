const { Router } = require("express");
const userRouter = Router();
const { userModel } = require("../db");
const bcrypt = require("bcrypt");
const { z } = require("zod");
const jwt = require("jsonwebtoken");
const { userMiddleware } = require("../middleware/user");


const {JWT_USER_SECRET} = require("../config")

userRouter.post('/signup', async (req, res) => {
    const { email, password, firstName, lastName } = req.body;

    const reqBody = z.object({
        email: z.string().email(),
        password: z.string(),
        firstName: z.string(),
        lastName: z.string()
    });

    const parsed = reqBody.safeParse(req.body);
    if (!parsed.success) {
        return res.json({ message: "Incorrect format" });
    }

    const hashedpass = await bcrypt.hash(password, 5);

    await userModel.create({ email, password: hashedpass, firstName, lastName });

    res.json({ message: "User registered" });
});

userRouter.post('/login', async (req, res) => {
    const { email, password } = req.body;
    const response = await userModel.findOne({ email });

    if (!response) {
        return res.status(404).json({ message: "User not found" });
    }

    const passMatch = await bcrypt.compare(password, response.password);

    if (passMatch) {
        const token = jwt.sign({ id: response._id }, JWT_USER_SECRET);
        return res.json({ token });
    } else {
        return res.status(401).json({ message: "Incorrect password" });
    }
});

userRouter.get('/userprofile', userMiddleware, async (req, res) => {
    const { userId } = req;
    const response = await userModel.findById(userId);
    if (!response) {
        return res.status(404).json({ message: "User not found" });
    }
    res.json(response);
});

module.exports = {
    userRouter
};