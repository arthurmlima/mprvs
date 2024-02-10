----------------------------------------------------------------------------------
-- Company: 
-- Engineer: 
-- 
-- Create Date: 08/29/2019 07:35:14 PM
-- Design Name: 
-- Module Name: wrapper_EB_DEC - Behavioral
-- Project Name: 
-- Target Devices: 
-- Tool Versions: 
-- Description: 
-- 
-- Dependencies: 
-- 
-- Revision:
-- Revision 0.01 - File Created
-- Additional Comments:
-- 
----------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.std_logic_unsigned.all;




entity N_wrapper_EB_DEC is
generic(
    x_init          : integer := 0;
    y_init          : integer := 0;
    img_width       : natural;
    img_height      : natural;
    n_frames        : natural;
    n_steps         : natural;
    pix_depth       : natural;
    subimg_width    : natural;
    subimg_height   : natural;
    buffer_length   : natural
);
  Port (
clk                        : in std_logic ;
reset                        : in std_logic ;
--reset_in_N_ctrl            : in std_logic;
--------Handshakes da entrada do EB
--P_N_valid_CF_EB              : in    std_logic; -- indica msg dispon?vel pra enviar ao EB
P_N_ready_EB_CF              : out   std_logic; -- indica que o EB pode receber a mensagem 
P_N_stall_in_EB              : in    std_logic ; --- indica que o processo deve ser stall para que exista sincronia dos elementos -> no CF
P_N_stall_out_EB             : inout std_logic;  -- indica que o '''
---------handshakes na sa?da do decoder
P_N_out_send                 : out  std_logic;
--P_N_ready_CF_DEC             : in std_logic;
P_N_stall_out_DEC            : inout std_logic ;
P_N_stall_in_DEC             : in std_logic ;
-------------------------------------------
--P_N_in_ack : in std_logic;

P_N_data_in_pixel     : in std_logic_vector(pix_depth-1 downto 0);
P_N_data_in_x_dest    : in std_logic_vector(img_width-1 downto 0);
P_N_data_in_y_dest    : in std_logic_vector(img_height-1 downto 0);
P_N_data_in_step      : in std_logic_vector(n_steps-1 downto 0);
P_N_data_in_frame     : in std_logic_vector(n_frames-1 downto 0);
P_N_data_in_x_orig    : in std_logic_vector(img_width-1 downto 0);
P_N_data_in_y_orig    : in std_logic_vector(img_height-1 downto 0);
P_N_data_in_fb        : in std_logic;
 
P_N_data_out_pixel     : out std_logic_vector(pix_depth-1 downto 0);    
P_N_data_out_x_dest    : out std_logic_vector(img_width-1 downto 0);    
P_N_data_out_y_dest    : out std_logic_vector(img_height-1 downto 0);   
P_N_data_out_step      : out std_logic_vector(n_steps-1 downto 0);      
P_N_data_out_frame     : out std_logic_vector(n_frames-1 downto 0);     
P_N_data_out_x_orig    : out std_logic_vector(img_width-1 downto 0);    
P_N_data_out_y_orig    : out std_logic_vector(img_height-1 downto 0);   
P_N_data_out_fb        : out std_logic   ;
--Wrapper_N_ready_EB_CF  : in std_logic;
P_N_out_direction      : out std_logic_vector(5 downto 0);
P_N_ack                : out std_logic;
P_N_req                : in std_logic
   
  
   );
end N_wrapper_EB_DEC;

architecture Behavioral of N_wrapper_EB_DEC is

signal	   w_N_ready_DEC_EB             :  std_logic; -- sinal indica se o decoder est? pronto para receber a mensagem 

signal     w_N_x_dest                   :  std_logic_vector(img_width-1 downto 0);
signal     w_N_y_dest                   :  std_logic_vector(img_height-1 downto 0);
signal     w_N_x_orig                   :  std_logic_vector(img_width-1 downto 0);
signal     w_N_y_orig                   :  std_logic_vector(img_height-1 downto 0);
signal     w_N_fb                       :  std_logic;
signal     w_N_out_send                 :  std_logic;
signal     w_N_ready_CF_DEC             :  std_logic;
signal     w_N_out_direction            :  std_logic_vector(5 downto 0);

SIGnal     w_N_valid_EB_DEC   :  std_logic;
SIGnal     w_N_out_new_EB_DEC  :  std_logic;

signal    s_N_data_pixel     :  std_logic_vector(pix_depth-1 downto 0);  
signal    s_N_data_x_dest    :  std_logic_vector(img_width-1 downto 0);  
signal    s_N_data_y_dest    :  std_logic_vector(img_height-1 downto 0); 
signal    s_N_data_step      :  std_logic_vector(n_steps-1 downto 0);    
signal    s_N_data_frame     :  std_logic_vector(n_frames-1 downto 0);   
signal    s_N_data_x_orig    :  std_logic_vector(img_width-1 downto 0);  
signal    s_N_data_y_orig    :  std_logic_vector(img_height-1 downto 0); 
signal    s_N_data_fb        :  std_logic   ;                                                                    
signal    s_N_ack                :  std_logic;                               
signal    s_N_req                : std_logic;                              





      
begin

Inst_dec : entity work.N_dec

generic map(
                                    x_init          => x_init       ,
                                    y_init          => y_init       ,
                                    img_width       => img_width    ,
                                    img_height      => img_height   ,
                                    n_frames        => n_frames     ,
                                    n_steps         => n_steps      ,
                                    pix_depth       => pix_depth    ,
                                    subimg_width    => subimg_width ,
                                    subimg_height   => subimg_height,
                                    buffer_length   => buffer_length
                                    )

port map (


clk                   =>  clk ,
rst                   => reset ,



N_stall_out_DEC      =>   P_N_stall_out_DEC ,
N_stall_in_DEC       =>   P_N_stall_in_DEC  ,

N_in_x_dest          =>  w_N_x_dest,
N_in_y_dest          =>  w_N_y_dest,
N_in_x_orig          =>  w_N_x_orig,
N_in_y_orig          =>  w_N_y_orig,
N_in_fb              =>  w_N_fb    ,

N_out_direction      =>  P_N_out_direction,

N_out_send           => P_N_out_send   ,


N_ready_DEC_EB       => w_N_ready_DEC_EB,
N_valid_EB_DEC       => w_N_valid_EB_DEC,
N_out_new_EB_DEC     => w_N_out_new_EB_DEC

);

Inst_elastic_buffer : entity work.N_elastic_buffer

generic map(
x_init          => x_init       ,
y_init          => y_init       ,
img_width       => img_width    ,
img_height      => img_height   ,
n_frames        => n_frames     ,
n_steps         => n_steps      ,
pix_depth       => pix_depth    ,
subimg_width    => subimg_width ,
subimg_height   => subimg_height,
buffer_length   => buffer_length
)
port map (


clk                     =>     clk , 
rst                     =>     reset ,       
          
N_valid_EB_DEC         =>     w_N_valid_EB_DEC,
N_ready_DEC_EB         =>     w_N_ready_DEC_EB,
  
N_out_new_EB_DEC     => w_N_out_new_EB_DEC,  

N_ready_EB_CF          =>  P_N_ready_EB_CF,
    
N_stall_in_EB          => P_N_stall_in_EB,
N_stall_out_EB         => P_N_stall_out_EB,

                       
N_data_in_pixel        =>    s_N_data_pixel  ,
N_data_in_x_dest       =>    s_N_data_x_dest ,
N_data_in_y_dest       =>    s_N_data_y_dest ,
N_data_in_step         =>    s_N_data_step   ,
N_data_in_frame        =>    s_N_data_frame  ,
N_data_in_x_orig       =>    s_N_data_x_orig ,
N_data_in_y_orig       =>    s_N_data_y_orig ,
N_data_in_fb           =>    s_N_data_fb     ,
N_ack                  =>    s_N_ack         ,
N_req                  =>    s_N_req         ,

N_data_out_pixel       =>     P_N_data_out_pixel ,
N_data_out_x_dest      =>     w_N_x_dest,
N_data_out_y_dest      =>     w_N_y_dest,
N_data_out_step        =>     P_N_data_out_step  ,
N_data_out_frame       =>     P_N_data_out_frame ,
N_data_out_x_orig      =>     w_N_x_orig,
N_data_out_y_orig      =>     w_N_y_orig,
N_data_out_fb          =>     w_N_fb  
--N_ready_CF_EB          =>     Wrapper_N_ready_EB_CF,






   


);


Inst_in_N_ctrl : entity work.in_N_controller

generic map(
x_init          => x_init       ,
y_init          => y_init       ,
img_width       => img_width    ,
img_height      => img_height   ,
n_frames        => n_frames     ,
n_steps         => n_steps      ,
pix_depth       => pix_depth    
)
port map (


clk                     =>     clk, 
reset                   =>     reset,       
                                 
oc_N_pixel        =>    P_N_data_in_pixel  ,
oc_N_x_dest       =>    P_N_data_in_x_dest ,
oc_N_y_dest       =>    P_N_data_in_y_dest ,
oc_N_step         =>    P_N_data_in_step   ,
oc_N_frame        =>    P_N_data_in_frame  ,
oc_N_x_orig       =>    P_N_data_in_x_orig,
oc_N_y_orig       =>    P_N_data_in_y_orig,
oc_N_fb           =>    P_N_data_in_fb    ,        
oc_N_new_msg      =>    P_N_req,
oc_N_ack          =>    P_N_ack, 


    

 i_N_pixel       =>     s_N_data_pixel  ,
 i_N_x_dest      =>     s_N_data_x_dest ,
 i_N_y_dest      =>     s_N_data_y_dest ,
 i_N_step        =>     s_N_data_step   ,
 i_N_frame       =>     s_N_data_frame  ,
 i_N_x_orig      =>     s_N_data_x_orig ,
 i_N_y_orig      =>     s_N_data_y_orig ,
 i_N_fb          =>     s_N_data_fb     ,
 i_N_req         =>     s_N_req         ,
 i_N_ack         =>     s_N_ack          




   


);







     P_N_data_out_x_dest <= w_N_x_dest;
     P_N_data_out_y_dest <= w_N_y_dest;
     
     P_N_data_out_x_orig <= w_N_x_orig;
     P_N_data_out_y_orig <= w_N_y_orig;
     P_N_data_out_fb     <= w_N_fb    ;
     
 

   
end Behavioral;
