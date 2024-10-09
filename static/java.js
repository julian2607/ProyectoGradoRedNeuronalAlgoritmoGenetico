// Alertas
const alertPlaceholder = document.getElementById('liveAlertPlaceholder')

const appendAlert = (message, type) => {
const wrapper = document.createElement('div')
wrapper.innerHTML = [
    `<div class="alert alert-${type} alert-dismissible" role="alert">`,
    `   <div>${message}</div>`,
    '   <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>',
    '</div>'
].join('')
alertPlaceholder.append(wrapper)
}

const alertTrigger = document.getElementById('liveAlertBtn')
if (alertTrigger) {
alertTrigger.addEventListener('click', () => {
    appendAlert('Nice, you triggered this alert message!', 'success')
})
}


// ACTUALIZAR MODELO
angular.module('myApp', [])
        .controller('myCtrl', function ($scope, $http, $timeout) {
            $scope.control=true;            
        // Actualizar Front
        function actualizarContador() {  
            $scope.control=false;      
            fetch('/ActualizarInformacion')
                .then(response => response.json())
                .then(data => {

                    //Datos reducicon dimensionalidad
                    try{
                        var tablaHtml = data.TablaPCA;
                        document.getElementById('TablaInfoReduccion').innerHTML = tablaHtml; 
                    }
                    catch{}

                    //Datos Entrenamiento
                    try{
                        var tablaHtml1 = data.contador;
                        document.getElementById('TablaInfo').innerHTML = tablaHtml1;                                      
                    }catch{}                    
                    
                });                        
        }
        setInterval(actualizarContador, 1000);
});

// PETICION PREPROCESAMIENTO
document.getElementById('RealizarPreProcesamiento').addEventListener('submit', function(event) {
    event.preventDefault(); 

    var loadingScreen = document.getElementById('loadingScreen');
    loadingScreen.classList.remove('hidden');
    var formData = new FormData(this);
    //MODAL RESULTADO
    var modal = document.getElementById('myModal');
    var modalMessage = document.getElementById('modalMessage');
    var closeModal = document.querySelector('.modal .close');

    fetch('http://127.0.0.1:5000/PreProcesamiento', {
        method: 'POST',
        body: formData
    })    
    .then(data => {
        loadingScreen.classList.add('hidden');
        modal.classList.remove('hidden')  
        modalMessage.textContent = "PreProcesamiento Realizado correctamente";   
        modal.style.display = 'flex';        
        console.log(data); 
    })
    .catch(error => {
        loadingScreen.classList.add('hidden');
        console.error('Error:', error);        
    });    
});

//REDUCCION DE DIMENSIONALIDAD
document.getElementById('PCAREDUCCION').addEventListener('submit', function(event) {
    event.preventDefault(); 

    //DATOS FROMUALRIO Y LOUDER
    var loadingScreen = document.getElementById('loadingScreen');
    loadingScreen.classList.remove('hidden');
    var formData = new FormData(this);

    //MODAL RESULTADO
    var modal = document.getElementById('myModal');
    var modalMessage = document.getElementById('modalMessage');
    // var closeModal = document.querySelector('.modal .close');

    fetch('http://127.0.0.1:5000/ReduccionDimensionalidadPCA', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        loadingScreen.classList.add('hidden');
        var tablaHtml = data.TablaPCA;
        document.getElementById('TablaInfoReduccion').innerHTML = tablaHtml;        
        modal.classList.remove('hidden')        
        appendAlert('Proceso realizado con exito', 'success');
        modalMessage.textContent = "PCA Realizado correctamente";         
        modal.style.display = 'flex';        
        console.log(data);
    })
    .catch(error => {
        loadingScreen.classList.add('hidden');
        console.error('Error:', error);        
    });

    //boton cerrar modal de aceptar
    closeModal.addEventListener('click', function() {
        appendAlert('Ha ocurrido un error entrenando el modelo', 'error');
        modal.style.display = 'none';
    });
});


//PETICION ENTRENAR SOLO MODELO
document.getElementById('EntrenarModeloAlgo').addEventListener('submit', function(event) {
    event.preventDefault(); 
    var loadingScreen = document.getElementById('loadingScreen');
    loadingScreen.classList.remove('hidden');
    var formData = new FormData(this);
    //Se agrega indicador para sber que boton hace la peticion
    formData.append('Indicador', '1');
    fetch('http://127.0.0.1:5000/ReduccionDimensionalidadPCA', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        loadingScreen.classList.add('hidden');
        // var tablaHtml = data.TablaPCA;
        // document.getElementById('TablaInfoReduccion').innerHTML = tablaHtml;
        modal.classList.remove('hidden')        
        appendAlert('Proceso realizado con exito', 'success');
        modalMessage.textContent = "Entrenamiento de la red realizada con exito.";         
        modal.style.display = 'flex';
        console.log(data);
    })
    .catch(error => {
        loadingScreen.classList.add('hidden');
        appendAlert('Ha ocurrido un error entrenando el modelo', 'error');
        console.error('Error:', error);        
    });
});


//Redirigir a nueva magian con botor
document.getElementById("Iniciar").addEventListener("click", function() {
    // Redirigir a otra página
    window.location.href = "http://127.0.0.1:5000/PaginaIniciar";
    loader.classList.add('hidden');
});

//PDF
document.getElementById("PDF").addEventListener("click", function() {
    // Redirigir a otra página
    window.location.href = "http://127.0.0.1:5000/PDF";
});

//EXCEL
document.getElementById("EXCEL").addEventListener("click", function() {
    // Redirigir a otra página
    window.location.href = "http://127.0.0.1:5000/EXCEL";
});
