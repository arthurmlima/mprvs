-- file: pixel_mem.vhd
-- description: pixel array per processPM_ing element
-- pixel PM_indexPM_ing - x coordPM_inate: column
--		  - y coordPM_inate: row
--                - PM_intermediary image PM_index: step
--                - current frame: frame

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;


entity PM_dec is

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
port(

	clk                         : in std_logic;
	rst                         : in std_logic;
	PM_ready_DEC_EB             : inout std_logic := '1'; --sinal indica se o decoder est? pronto para receber a mensagem 
	PM_stall_out_DEC            : inout std_logic := '0';
	PM_stall_in_DEC             : in std_logic ; 
    PM_in_x_dest                : in std_logic_vector(img_width-1 downto 0);
    PM_in_y_dest                : in std_logic_vector(img_height-1 downto 0);
    PM_in_x_orig                : in std_logic_vector(img_width-1 downto 0);
    PM_in_y_orig                : in std_logic_vector(img_height-1 downto 0);
    PM_in_fb                    : in std_logic;
    PM_out_send                 : out std_logic := '0';
    PM_out_direction            : out std_logic_vector(5 downto 0);
    PM_valid_EB_DEC             : in      std_logic;
    PM_out_new_EB_DEC           : in std_logic
    
);
end entity PM_dec;

architecture behavioral of PM_dec is

    --signal PM_out_direction : std_logic_vector(5 downto 0);--hotspot: PixMem, PE, N, S, E, W
    
    --signal pixel : natural := 0;
    signal x_dest: natural ;
    signal y_dest: natural ;
    signal x_dest_orig: natural ;
    signal y_dest_orig: natural ;
    --signal step  : natural := 0;
    --signal frame : natural := 0;
    signal x_orig: natural;
    signal y_orig: natural;
    --signal fb    : std_logic := '0';
    
    constant x_end : natural := x_init + subimg_width-1; -- end of region
    constant y_end : natural := y_init + subimg_height-1; -- end of region
    
    signal x_diff : integer;-- := x_dest-x_PM_init; -- distance to destPM_iny
    signal y_diff : integer;-- := y_dest-y_PM_init; -- distance to destPM_iny
    
    signal NW : std_logic := '0';
    signal SW : std_logic := '0';
    signal NE : std_logic := '0';
    signal SE : std_logic := '0';
    
    signal counter : natural := 0;
    signal state : std_logic := '0';
begin
    
    x_dest <= to_integer(unsigned(PM_in_x_dest));
    y_dest <= to_integer(unsigned(PM_in_y_dest));
    x_orig <= to_integer(unsigned(PM_in_x_orig));
    y_orig <= to_integer(unsigned(PM_in_y_orig));
    x_diff <= x_dest-x_init;
    y_diff <= y_dest-y_init;    

process(clk,rst) is
begin
    if(rst='1')then
        PM_out_direction <= "000000";

        PM_out_send    <= '0';
   
    else
       if(rising_edge(clk))then
       
       if(PM_out_new_EB_DEC = '1')then --
       state<= '0';
 --
       end if; 
       
       case state is 
       
       when '0' =>
            if(PM_out_new_EB_DEC = '1')then --  
            state<='1';
            PM_stall_out_DEC <= '1';
            
                   if(PM_in_fb='0')then

                           if( (x_dest>=x_init)and(x_dest<=x_end)and(y_dest>=y_init)and(y_dest<=y_end) ) then
                                
                                    PM_out_direction <= "100000"; 

                                   PM_out_send    <= '1';
                                   PM_ready_DEC_EB <= '0';

                                   
                           elsif( (x_dest>=x_init)and(x_dest<=x_end) )then -- rPM_oute PM_in Y (North/SPM_outh)
                               -- the above condition says that the message is PM_in the same X range of the local tile
                               -- defPM_ine now if rPM_oute North or SPM_outh
                               if( y_dest<y_init )then -- rPM_oute North
                                   PM_out_direction <= "001000";

                                   PM_out_send    <= '1';  
                                   PM_ready_DEC_EB <= '0';                                    
                               else -- rPM_oute SPM_outh
                                    PM_out_direction <= "000100";    
        
                                    PM_out_send    <= '1';   
                                    PM_ready_DEC_EB <= '0';                                       
                               end if;
                           elsif( (y_dest>=y_init)and(y_dest<=y_end) )then -- rPM_oute PM_in X (East/West)
                                -- the above condition says that the message is PM_in the same Y range of the local tile
                                -- defPM_ine now if rPM_oute East or West
                                if( x_dest<x_init)then -- rPM_oute West
                                    PM_out_direction <= "000001";   
      
                                    PM_out_send    <= '1';      
                                    PM_ready_DEC_EB <= '0';                                    
                                else -- rPM_oute East
                                    PM_out_direction <= "000010";       
          
                                    PM_out_send    <= '1';          
                                    PM_ready_DEC_EB <= '0';                                    
                                end if;
                           elsif( (x_dest<x_init)and(y_dest<y_init) )then -- North-West region
                                --defPM_ine now if rPM_oute North or West first
                                if(NW='0')then -- rPM_oute North
                                    PM_out_direction <= "001000";

                                    PM_out_send    <= '1';      
                                    PM_ready_DEC_EB <= '0';                           
                                else -- rPM_oute West
                                    PM_out_direction <= "000001";   
      
                                    PM_out_send    <= '1';      
                                    PM_ready_DEC_EB <= '0';                          
                                end if;
                                NW <= not(NW);
                           elsif( (x_dest<x_init)and(y_dest>y_end) )then -- SPM_outh-West region
                                -- defPM_ine now if rPM_oute SPM_outh or West first
                                if(SW='0')then -- rPM_oute SPM_outh
                                    PM_out_direction <= "000100";    
         
                                    PM_out_send    <= '1';      
                                    PM_ready_DEC_EB <= '0';                                 
                                else -- rPM_oute West
                                    PM_out_direction <= "000001";   
         
                                    PM_ready_DEC_EB <= '0';                       
                                end if;       
                                SW <= not(SW);                    
                           elsif( (x_dest>x_end)and(y_dest<y_init) )then -- North-East region
                                -- defPM_ine now if rPM_oute North or East first
                                if(NE='0')then -- rPM_oute North
                                    PM_out_direction <= "001000";

                                    PM_out_send    <= '1';                                 
                                    PM_ready_DEC_EB <= '0';
                                else -- rPM_oute East
                                    PM_out_direction <= "000010";       
          
                                    PM_out_send    <= '1';    
                                    PM_ready_DEC_EB <= '0';                              
                                end if;
                                NE <= not(NE);
                           elsif( (x_dest>x_end)and(y_dest>y_end) )then -- SPM_outh-East region
                                -- defPM_ine now if rPM_oute SPM_outh or East first
                                if(SE='0')then -- rPM_oute SPM_outh
                                    PM_out_direction <= "000100";    
        
                                    PM_out_send    <= '1';              
                                    PM_ready_DEC_EB <= '0';                         
                                else  -- rPM_oute East
                                    PM_out_direction <= "000010";       
           
                                    PM_out_send    <= '1';          
                                    PM_ready_DEC_EB <= '0';                        
                                end if;
                                SE <= not(SE);
                           end if;
                           
                       else
                      
                       --------------- FB = 1 = backward message
                        --PM_out_send <= '0';

                          if( (x_orig>=x_init)and(x_orig<=x_end)and(y_orig>=y_init)and(y_orig<=y_end) )then
--            
                                   PM_out_direction <= "010000"; -- rPM_oute to ProcessPM_ing Element

                                   PM_out_send    <= '1';   
                                   PM_ready_DEC_EB <= '0';                         
                           
                                  
                          elsif( (x_orig>=x_init)and(x_orig<=x_end) )then
                              -- the above condition says that the message is PM_in the same X range of the local tile
                              -- defPM_ine now if rPM_oute North or SPM_outh
                              if( y_orig<y_init )then -- rPM_oute North
                                  PM_out_direction <= "001000";

                                  PM_out_send    <= '1';     
                                  PM_ready_DEC_EB <= '0';                               
                              else -- rPM_oute SPM_outh
                                   PM_out_direction <= "000100";    
         
                                   PM_out_send    <= '1';         
                                   PM_ready_DEC_EB <= '0';                                
                              end if;
                          elsif( (y_orig>=y_init)and(y_orig<=y_end) )then -- rPM_oute PM_in X (East/West)
                               -- the above condition says that the message is PM_in the same Y range of the local tile
                               -- defPM_ine now if rPM_oute East or West
                               if( x_orig<x_init)then -- rPM_oute West
                                   PM_out_direction <= "000001";   
        
                                   PM_out_send    <= '1';           
                                   PM_ready_DEC_EB <= '0';                               
                               else -- rPM_oute East
                                   PM_out_direction <= "000010";       
            
                                   PM_out_send    <= '1';             
                                   PM_ready_DEC_EB <= '0';                                 
                               end if;
                          elsif( (x_orig<x_init)and(y_orig<y_init) )then -- North-West region
                               --defPM_ine now if rPM_oute North or West first
                               if(NW='0')then -- rPM_oute North
                                   PM_out_direction <= "001000";

                                   PM_out_send    <= '1';         
                                   PM_ready_DEC_EB <= '0';                        
                               else -- rPM_oute West
                                   PM_out_direction <= "000001";   
       
                                   PM_out_send    <= '1';     
                                   PM_ready_DEC_EB <= '0';                          
                               end if;
                               NW <= not(NW);
                          elsif( (x_orig<x_init)and(y_orig>y_end) )then -- SPM_outh-West region
                               -- defPM_ine now if rPM_oute SPM_outh or West first
                               if(SW='0')then -- rPM_oute SPM_outh
                                   PM_out_direction <= "000100";    
         
                                   PM_out_send    <= '1';     
                                   PM_ready_DEC_EB <= '0';                                 
                               else -- rPM_oute West
                                   PM_out_direction <= "000001";   
        
                                   PM_out_send    <= '1';         
                                   PM_ready_DEC_EB <= '0';                       
                               end if;       
                               SW <= not(SW);                    
                          elsif( (x_orig>x_end)and(y_orig<y_init) )then -- North-East region
                               -- defPM_ine now if rPM_oute North or East first
                               if(NE='0')then -- rPM_oute North
                                   PM_out_direction <= "001000";

                                   PM_out_send    <= '1';                                 
                                   PM_ready_DEC_EB <= '0';
                               else -- rPM_oute East
                                   PM_out_direction <= "000010";       
            
                                   PM_out_send    <= '1';        
                                   PM_ready_DEC_EB <= '0';                        
                               end if;
                               NE <= not(NE);
                          elsif( (x_orig>x_end)and(y_orig>y_end) )then -- SPM_outh-East region
                               -- defPM_ine now if rPM_oute SPM_outh or East first
                               if(SE='0')then -- rPM_oute SPM_outh
                                   PM_out_direction <= "000100";    
        
                                   PM_out_send    <= '1';    
                                   PM_ready_DEC_EB <= '0';                                
                               else -- rPM_oute East
                                   PM_out_direction <= "000010";       
            
                                   PM_out_send    <= '1';    
                                    PM_ready_DEC_EB <= '0';                              
                               end if;
                               SE <= not(SE);
                          end if;
                       
                       
                       end if;-- end fb
                   --end if;
                 --  counter <= 1;
                   
               end if;--PM_in ack=0 and PM_in_new=1
       when '1' =>

                if(PM_stall_in_DEC = '1')then
                
                    PM_ready_DEC_EB <= '1'; -- indica para o EB  que o decoder est? livre para receber msg
                    state<='0';
                    PM_out_send <= '0';
                    PM_stall_out_DEC <= '0';                   
                end if;
                
           when others =>
           
           end case;
        end if;
            
    end if;
end process;


end architecture behavioral;
