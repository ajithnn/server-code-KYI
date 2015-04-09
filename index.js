var rest = require('exprestify')
var fs = require('fs')
var exec = require('child_process').exec;

var header = {
    "Access-Control-Allow-Origin": "null",
    "Access-Control-Allow-Methods": "GET,PUT,POST,DELETE",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Credentials": "true"
};

var multiopt = {
    FilePath: "./assets/",
    PostType: "file",
    Rename: function(fld, file) {
        return "CurrentImg";
    }
};

rest.setHeaders(header);

rest.getfile('/image', function(err, query) {
    if (!err) {
        return "./assets/outImage_" + query.id + ".jpg";
    } else {
        console.log(err);
        return err;
    }
})

rest.get('/runpy', function(err, query,ctype) {
    if (!err) {
        var pathforPython = 'python ';
        var pathForFile = __dirname + '/python/grayFaceGreenEye.py ';
        exec(pathforPython + pathForFile + __dirname + " outImage_"+ query.id + ".jpg" , function(error, stdout, stderr) {
             console.log(stdout);  
              console.log(error);      
        });
        return "done"
    } else {
        console.log(err);
        return err;
    }
})

rest.multipost('/PostPhoto', function(err, data) {
    if (!err) {
        console.log(data);
        return "done";
    } else {
        console.log(err);
    }
}, multiopt);

var server = rest.getSocketServer()
var io = require('socket.io')(server)

io.on('connection', function(socket) {
	console.log("Connected: " + socket.id);
    fs.watch('./assets/', function(event, filename) {
        console.log(filename);
        if(filename == "outImage_"+socket.id+".jpg")
        {
        consle.log("Inside Emitter");
        var time = new Date().getTime();
        socket.to(socket.id).emit("ImageModified", "/image?id=" + socket.id + "&time=" + time);
        }
    });
});
rest.port = process.env.PORT || 3000 ;
server.listen(process.env.PORT || 3000, function() {
    console.log("Listening on port 0.0.0.0:%s", process.env.PORT || rest.port)
})
