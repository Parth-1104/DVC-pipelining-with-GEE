const mongoose = require("mongoose");
const Schema = mongoose.Schema;
const ObjectId = mongoose.Types.ObjectId;

const userSchema = new Schema({
    id : ObjectId,
    email : {type : String, unique: true},
    password : String,
    firstName : String,
    lastName : String

})
const adminSchema = new Schema({
    id : ObjectId,
    email : {type : String, unique: true},
    password : String,
    firstName : String,
    lastName : String

})

const adminModel = mongoose.model('admins', adminSchema)
const userModel = mongoose.model('users', userSchema)



module.exports= {
    userModel,
    adminModel
}