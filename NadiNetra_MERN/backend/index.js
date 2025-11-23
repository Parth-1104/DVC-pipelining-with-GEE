const express = require('express');
const app = express();
require('dotenv').config();
const port =  3000;
const mongoose = require('mongoose');
const cors = require('cors');
app.use(cors());
app.use(express.json());
const {userRouter} = require('./routes/user');
const {adminRouter} = require('./routes/admin');

app.use('/api/v1/user', userRouter);
app.use('/api/v1/admin', adminRouter);

app.get('/', (req, res) => {
  res.send('Hello World!');
});
async function main(){
  await mongoose.connect(process.env.MONGO_URL)
  app.listen(port, () => {
    console.log(`Example app listening at http:/user//user/localhost:${port}`);
  });
}
main()